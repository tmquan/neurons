"""
Cosmos-Predict2.5 **2D** model wrapper for connectomics segmentation.

Adapts the Cosmos-Predict2.5 Diffusion Transformer backbone (2B or 14B
variant) as a feature extractor for the three-stage 2D segmentation task:

- **Semantic**: per-pixel class logits  (``num_classes`` channels)
- **Instance**: per-pixel embedding vectors  (``emb_dim`` channels)
- **Geometry**: per-pixel direction, covariance, and RGBA reconstruction

The backbone is loaded from HuggingFace Hub with automatic caching.
Both the VAE encoder (tokenizer) and DiT transformer are used:

1. Input images are adapted from single-channel EM to 3-channel RGB.
2. The VAE encoder compresses to a latent grid (spatial_compression x).
3. DiT transformer blocks process the latents; intermediate features are
   extracted from configurable layers.
4. A progressive upsampler restores spatial resolution.
5. Three parallel task heads produce dense predictions.

References:
    - https://github.com/nvidia-cosmos/cosmos-predict2.5
    - HuggingFace: nvidia/Cosmos-Predict2.5-2B, nvidia/Cosmos-Predict2.5-14B
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, einsum

from neurons.models.point_prompt_encoder import PointPromptEncoder

logger = logging.getLogger(__name__)

_SPATIAL_DIMS = 2
_CONV = nn.Conv2d


def _NORM(ch: int) -> nn.GroupNorm:
    num_groups = max(g for g in (1, 2, 4, 8, 16, 32) if ch % g == 0)
    return nn.GroupNorm(num_groups, ch)


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------

@dataclass
class _VariantConfig:
    """Architecture and download metadata for a Cosmos-Predict2.5 variant."""

    hf_repo_id: str
    hf_revision: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    latent_channels: int
    spatial_compression: int
    temporal_compression: int
    estimated_vram_gb: float
    max_sequence_length: int
    patch_size: int = 2
    mlp_ratio: float = 4.0


_VARIANT_CONFIGS: Dict[str, _VariantConfig] = {
    "2B": _VariantConfig(
        hf_repo_id="nvidia/Cosmos-Predict2.5-2B",
        hf_revision="diffusers/base/post-trained",
        hidden_dim=2048,
        num_layers=28,
        num_heads=16,
        latent_channels=16,
        spatial_compression=8,
        temporal_compression=8,
        estimated_vram_gb=8.0,
        max_sequence_length=32768,
    ),
    "14B": _VariantConfig(
        hf_repo_id="nvidia/Cosmos-Predict2.5-14B",
        hf_revision="diffusers/base/post-trained",
        hidden_dim=5120,
        num_layers=40,
        num_heads=40,
        latent_channels=16,
        spatial_compression=8,
        temporal_compression=8,
        estimated_vram_gb=32.0,
        max_sequence_length=32768,
    ),
}


# ---------------------------------------------------------------------------
# HuggingFace weight utilities
# ---------------------------------------------------------------------------

def _download_from_hf(
    repo_id: str,
    revision: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Download model snapshot from HuggingFace Hub.

    Returns the local path to the downloaded snapshot directory.
    Handles authentication, caching, and connection failures.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading Cosmos-Predict2.5 "
            "weights.  Install with: pip install huggingface_hub"
        )

    cache_dir = cache_dir or str(Path.home() / ".cache" / "neurons" / "cosmos_predict25")

    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            ignore_patterns=["*.md", "*.txt", "*.json", "examples/*", "docs/*"],
        )
        logger.info("Downloaded %s (rev=%s) -> %s", repo_id, revision, local_path)
        return Path(local_path)
    except Exception as e:
        logger.warning(
            "HuggingFace download failed for %s (rev=%s): %s. "
            "Falling back to random initialization.",
            repo_id, revision, e,
        )
        raise


# ---------------------------------------------------------------------------
# Input channel adapter
# ---------------------------------------------------------------------------

class _InputAdapter(nn.Module):
    """Project arbitrary input channels to 3-ch RGB expected by Cosmos."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        if in_channels == 3:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Lightweight DiT block (used when diffusers is unavailable)
# ---------------------------------------------------------------------------

class _DiTBlock(nn.Module):
    """Single Diffusion-Transformer block with adaptive layer norm.

    Simplified version of the Cosmos-Predict2.5 DiT block for standalone
    use when the ``diffusers`` or ``cosmos_predict2`` packages are not
    installed.  Matches the hidden_dim / num_heads so that weights loaded
    from the checkpoint have compatible shapes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if timestep_emb is not None:
            modulation = self.adaLN_modulation(timestep_emb)
            shift1, scale1, shift2, scale2 = rearrange(
                modulation, "b (n d) -> n b d", n=4,
            ).unbind(0)
            h = self.norm1(x) * (1 + rearrange(scale1, "b d -> b 1 d")) + rearrange(shift1, "b d -> b 1 d")
        else:
            h = self.norm1(x)

        h, _ = self.attn(h, h, h)
        x = x + h

        if timestep_emb is not None:
            h = self.norm2(x) * (1 + rearrange(scale2, "b d -> b 1 d")) + rearrange(shift2, "b d -> b 1 d")
        else:
            h = self.norm2(x)

        return x + self.mlp(h)


class _StandaloneDiT(nn.Module):
    """Minimal Diffusion Transformer matching Cosmos-Predict2.5 shape.

    Used as a drop-in when the official ``diffusers`` pipeline is not
    available.  The architecture reproduces the core structure (patch
    embed -> N x DiTBlock -> unpatchify) so that pretrained weights can be
    loaded by name when available.
    """

    def __init__(self, cfg: _VariantConfig) -> None:
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.patch_size = cfg.patch_size
        self.latent_channels = cfg.latent_channels

        patch_input_dim = cfg.latent_channels * cfg.patch_size * cfg.patch_size
        self.patch_embed = nn.Linear(patch_input_dim, cfg.hidden_dim)

        self.timestep_embed = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.blocks = nn.ModuleList([
            _DiTBlock(cfg.hidden_dim, cfg.num_heads, cfg.mlp_ratio)
            for _ in range(cfg.num_layers)
        ])
        self.norm_out = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        feature_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Run DiT and optionally return intermediate features.

        Args:
            latent: Latent tensor ``[B, C_lat, H_lat, W_lat]``.
            timestep: Scalar or ``[B]`` timestep for adaptive norm.
            feature_layers: Block indices whose outputs to capture.

        Returns:
            (final_hidden, intermediate_features) where
            ``final_hidden`` is ``[B, N, hidden_dim]`` and
            ``intermediate_features`` maps layer index -> ``[B, N, D]``.
        """
        _param_dtype = self.patch_embed.weight.dtype
        latent = latent.to(dtype=_param_dtype)

        B, C, H, W = latent.shape
        P = self.patch_size

        patches = rearrange(
            latent,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=P, p2=P,
        )
        x = self.patch_embed(patches)

        if timestep is not None:
            if timestep.dim() == 0:
                timestep = repeat(timestep, "-> b 1", b=B)
            elif timestep.dim() == 1:
                timestep = rearrange(timestep, "b -> b 1")
            t_emb = self.timestep_embed(timestep.to(dtype=_param_dtype))
        else:
            t_emb = None

        feature_layers = feature_layers or []
        intermediates: Dict[int, torch.Tensor] = {}

        for idx, block in enumerate(self.blocks):
            x = block(x, t_emb)
            if idx in feature_layers:
                intermediates[idx] = x

        x = self.norm_out(x)
        return x, intermediates


# ---------------------------------------------------------------------------
# Feature pyramid + upsampler
# ---------------------------------------------------------------------------

class _FeatureProjector(nn.Module):
    """Fuse multi-layer DiT features into a single spatial feature map.

    Accepts features from several DiT layers (each ``[B, N, D]``), reshapes
    them to spatial grids, concatenates along channel dim, and projects to
    ``out_dim``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_feature_layers: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        total_in = hidden_dim * num_feature_layers
        self.proj = nn.Sequential(
            _CONV(total_in, out_dim * 2, 1),
            _NORM(out_dim * 2),
            nn.GELU(),
            _CONV(out_dim * 2, out_dim, 1),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        h: int,
        w: int,
    ) -> torch.Tensor:
        spatial = [
            rearrange(f, "b (h w) d -> b d h w", h=h, w=w)
            for f in features
        ]
        fused = torch.cat(spatial, dim=1)
        return self.proj(fused)


class _ProgressiveUpsampler(nn.Module):
    """Upsample latent feature grid to target spatial resolution.

    Uses ``log2(scale_factor)`` stages of transposed convolution,
    each doubling spatial size.
    """

    def __init__(self, in_dim: int, out_dim: int, num_stages: int) -> None:
        super().__init__()
        dims = self._interpolate_dims(in_dim, out_dim, num_stages + 1)
        layers: List[nn.Module] = []
        for i in range(num_stages):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(dims[i], dims[i + 1], 4, stride=2, padding=1),
                _NORM(dims[i + 1]),
                nn.GELU(),
            ))
        self.stages = nn.ModuleList(layers)

    @staticmethod
    def _interpolate_dims(start: int, end: int, n: int) -> List[int]:
        """Linearly interpolate channel counts, ensuring divisibility by 8."""
        if n <= 1:
            return [start]
        step = (end - start) / (n - 1)
        dims = [max(8, int(round((start + i * step) / 8)) * 8) for i in range(n)]
        dims[0], dims[-1] = start, end
        return dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CosmosPredict2DWrapper(nn.Module):
    """Cosmos-Predict2.5 adapted for 2D connectomics segmentation.

    Three parallel output heads produce:

    - ``semantic``  [B, num_classes, H, W]
    - ``instance``  [B, emb_dim, H, W]
    - ``geometry``  [B, G, H, W]  where G = S + S**2 + 4 (dir + cov + rgba)

    Args:
        in_channels: Number of input channels (1 for EM images).
        num_classes: Semantic segmentation classes.
        emb_dim: Instance embedding dimensionality.
        feature_size: Internal feature map channel count after projection.
        variant: ``"2B"`` or ``"14B"`` model variant.
        checkpoint_variant: HuggingFace revision string.
        dtype: Weight dtype (``"bf16"``, ``"fp16"``, ``"fp32"``).
        freeze_backbone: Whether to freeze the pretrained DiT backbone.
        feature_layers: DiT block indices to extract features from.
            Defaults to quartile layers for the chosen variant.
        cache_dir: HuggingFace download cache directory.
        hf_token: HuggingFace authentication token.
        dropout: Dropout probability for heads.

    Example::

        >>> model = CosmosPredict2DWrapper(
        ...     in_channels=1, num_classes=16, variant="2B",
        ... )
        >>> x = torch.randn(2, 1, 256, 256)
        >>> out = model(x)
        >>> out["semantic"].shape   # [2, 16, 256, 256]
        >>> out["instance"].shape   # [2, 16, 256, 256]
        >>> out["geometry"].shape   # [2, 10, 256, 256]
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 16,
        emb_dim: int = 16,
        feature_size: int = 64,
        variant: str = "2B",
        checkpoint_variant: str = "post-trained",
        dtype: str = "bf16",
        freeze_backbone: bool = False,
        feature_layers: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        variant = variant.upper()
        if variant not in _VARIANT_CONFIGS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(_VARIANT_CONFIGS)}"
            )

        self.variant = variant
        self.cfg = _VARIANT_CONFIGS[variant]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.feature_size = feature_size
        self.spatial_dims = _SPATIAL_DIMS
        self.dropout = dropout
        self.geom_channels = _SPATIAL_DIMS + _SPATIAL_DIMS ** 2 + 4

        self._dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self._freeze_backbone = freeze_backbone

        if feature_layers is not None:
            self._feature_layers = sorted(feature_layers)
        else:
            n = self.cfg.num_layers
            self._feature_layers = sorted({n // 4, n // 2, 3 * n // 4, n - 1})

        self.input_adapter = _InputAdapter(in_channels)

        self._backbone_loaded = False
        self._build_backbone(cache_dir, hf_token, checkpoint_variant)

        num_upsample_stages = int(math.log2(self.cfg.spatial_compression))

        self.feature_projector = _FeatureProjector(
            hidden_dim=self.cfg.hidden_dim,
            num_feature_layers=len(self._feature_layers),
            out_dim=feature_size,
        )

        self.upsampler = _ProgressiveUpsampler(
            in_dim=feature_size,
            out_dim=feature_size,
            num_stages=num_upsample_stages,
        )

        self.head_semantic = nn.Sequential(
            _CONV(feature_size, 64, 3, padding=1), _NORM(64), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            _CONV(64, num_classes, 1),
        )
        self.head_instance = nn.Sequential(
            _CONV(feature_size, 64, 3, padding=1), _NORM(64), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            _CONV(64, emb_dim, 1),
        )
        self.head_geometry = nn.Sequential(
            _CONV(feature_size, 64, 3, padding=1), _NORM(64), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            _CONV(64, self.geom_channels, 1),
        )

        self.point_encoder = PointPromptEncoder(
            num_classes=num_classes,
            feature_size=feature_size,
            spatial_dims=_SPATIAL_DIMS,
        )

        if self.vae_encoder is not None:
            self.vae_encoder.requires_grad_(False)
            self.vae_encoder.eval()

        if freeze_backbone:
            self.freeze_encoder()

        logger.info(
            "CosmosPredict2DWrapper initialised: variant=%s, "
            "feature_layers=%s, backbone_loaded=%s, frozen=%s, "
            "params=%s (trainable=%s)",
            variant, self._feature_layers, self._backbone_loaded,
            freeze_backbone,
            f"{self.get_num_parameters(trainable_only=False):,}",
            f"{self.get_num_parameters(trainable_only=True):,}",
        )

    # ------------------------------------------------------------------
    # Backbone construction
    # ------------------------------------------------------------------

    def _build_backbone(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> None:
        """Build or load the DiT backbone via one of three strategies."""
        self.vae_encoder: Optional[nn.Module] = None
        self.dit: nn.Module

        loaded = (
            self._try_load_diffusers(cache_dir, hf_token, checkpoint_variant)
            or self._try_load_cosmos_package(cache_dir, hf_token, checkpoint_variant)
            or self._try_load_raw_checkpoint(cache_dir, hf_token, checkpoint_variant)
        )

        if not loaded:
            logger.warning(
                "No pretrained weights loaded -- using randomly initialised "
                "DiT backbone (%s architecture).  This is expected if running "
                "without network access or HuggingFace credentials.",
                self.variant,
            )
            self._build_standalone_backbone()

    def _try_load_diffusers(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        """Attempt to load backbone components via ``diffusers``."""
        try:
            from diffusers import AutoencoderKLCosmos, CosmosTransformer3DModel
        except ImportError:
            logger.debug("diffusers Cosmos classes not available.")
            return False

        try:
            revision = f"diffusers/base/{checkpoint_variant}"

            # TODO: The exact subfolder layout may vary between diffusers
            # versions.  Verify against the installed diffusers release.
            transformer = CosmosTransformer3DModel.from_pretrained(
                self.cfg.hf_repo_id,
                subfolder="transformer",
                revision=revision,
                torch_dtype=self._dtype,
                cache_dir=cache_dir,
                token=hf_token,
            )
            vae = AutoencoderKLCosmos.from_pretrained(
                self.cfg.hf_repo_id,
                subfolder="vae",
                revision=revision,
                torch_dtype=self._dtype,
                cache_dir=cache_dir,
                token=hf_token,
            )

            self.vae_encoder = vae.encoder
            self.vae_encoder.requires_grad_(False)
            self.vae_encoder.eval()

            self.dit = transformer
            self._backbone_loaded = True
            self._backend = "diffusers"
            logger.info("Loaded backbone via diffusers (revision=%s).", revision)
            return True

        except Exception as exc:
            logger.debug("diffusers load failed: %s", exc)
            return False

    def _try_load_cosmos_package(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        """Attempt to load via the ``cosmos_predict2`` package."""
        try:
            # TODO: Exact import path depends on the installed cosmos_predict2
            # version.  The package layout observed in the repo is:
            #   cosmos_predict2._src.predict2.models.dit
            from cosmos_predict2.inference import CosmosPredict2Pipeline  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("cosmos_predict2 package not available.")
            return False

        try:
            pipe = CosmosPredict2Pipeline.from_pretrained(
                self.cfg.hf_repo_id,
                cache_dir=cache_dir,
                token=hf_token,
            )
            # TODO: Extract VAE encoder and DiT from the pipeline object.
            # The exact attribute names need verification against the
            # installed cosmos_predict2 version.
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "encoder"):
                self.vae_encoder = pipe.vae.encoder
                self.vae_encoder.requires_grad_(False)
                self.vae_encoder.eval()

            if hasattr(pipe, "dit"):
                self.dit = pipe.dit
            elif hasattr(pipe, "transformer"):
                self.dit = pipe.transformer
            else:
                logger.warning("Could not find DiT module on cosmos_predict2 pipeline.")
                return False

            self._backbone_loaded = True
            self._backend = "cosmos_predict2"
            logger.info("Loaded backbone via cosmos_predict2 package.")
            return True

        except Exception as exc:
            logger.debug("cosmos_predict2 load failed: %s", exc)
            return False

    def _try_load_raw_checkpoint(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        """Download raw checkpoint and load into standalone DiT."""
        try:
            local_path = _download_from_hf(
                self.cfg.hf_repo_id,
                revision=f"diffusers/base/{checkpoint_variant}",
                cache_dir=cache_dir,
                token=hf_token,
            )
        except Exception:
            return False

        self._build_standalone_backbone()

        ckpt_files = list(local_path.glob("**/*.safetensors")) + list(local_path.glob("**/*.pt"))
        if not ckpt_files:
            logger.warning("No checkpoint files found in %s.", local_path)
            return False

        loaded_any = False
        for ckpt_file in ckpt_files:
            try:
                if ckpt_file.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    state = load_file(str(ckpt_file), device="cpu")
                else:
                    state = torch.load(str(ckpt_file), map_location="cpu", weights_only=True)

                missing, unexpected = self.dit.load_state_dict(state, strict=False)
                if missing:
                    logger.debug("Missing keys when loading %s: %d", ckpt_file.name, len(missing))
                if unexpected:
                    logger.debug("Unexpected keys in %s: %d", ckpt_file.name, len(unexpected))
                loaded_any = True
            except Exception as exc:
                logger.debug("Failed to load %s: %s", ckpt_file.name, exc)

        if loaded_any:
            self._backbone_loaded = True
            self._backend = "raw_checkpoint"
            logger.info("Loaded weights from raw checkpoint into standalone DiT.")

        return loaded_any

    def _build_standalone_backbone(self) -> None:
        """Construct a standalone DiT backbone with matching architecture."""
        self.dit = _StandaloneDiT(self.cfg)
        self._backend = "standalone"

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space image to latent grid.

        If a pretrained VAE encoder is available, uses it (in eval / no_grad).
        Otherwise, applies a learned conv-based downsampler.

        Args:
            x: RGB-adapted input ``[B, 3, H, W]``.

        Returns:
            Latent tensor ``[B, C_lat, H_lat, W_lat]`` where spatial dims
            are compressed by ``spatial_compression``.
        """
        if self.vae_encoder is not None:
            with torch.no_grad():
                if x.dim() == 4:
                    x_5d = rearrange(x, "b c h w -> b c 1 h w")
                else:
                    x_5d = x

                # TODO: The exact VAE encoder API may accept (x) or (x).sample.
                # Adjust after verifying the installed diffusers / cosmos version.
                enc = self.vae_encoder(x_5d)
                if hasattr(enc, "latent_dist"):
                    latent = enc.latent_dist.mode()
                elif hasattr(enc, "sample"):
                    latent = enc.sample
                else:
                    latent = enc

                if latent.dim() == 5:
                    latent = rearrange(latent, "b c 1 h w -> b c h w")
                return latent.to(dtype=x.dtype)

        return self._conv_downsample(x)

    def _conv_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback learned downsampler when no VAE is available."""
        if not hasattr(self, "_fallback_down"):
            s = self.cfg.spatial_compression
            self._fallback_down = nn.Sequential(
                _CONV(3, self.cfg.latent_channels * 2, kernel_size=s, stride=s),
                _NORM(self.cfg.latent_channels * 2),
                nn.GELU(),
                _CONV(self.cfg.latent_channels * 2, self.cfg.latent_channels, 1),
            ).to(x.device, x.dtype)
        return self._fallback_down(x)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, latent: torch.Tensor) -> torch.Tensor:
        """Run DiT backbone and extract multi-layer spatial features.

        Returns:
            Feature map ``[B, feature_size, H_lat, W_lat]``.
        """
        B, C, H_lat, W_lat = latent.shape
        P = self.cfg.patch_size

        pad_h = (P - H_lat % P) % P
        pad_w = (P - W_lat % P) % P
        if pad_h > 0 or pad_w > 0:
            latent = F.pad(latent, (0, pad_w, 0, pad_h), mode="reflect")
            H_lat_p, W_lat_p = H_lat + pad_h, W_lat + pad_w
        else:
            H_lat_p, W_lat_p = H_lat, W_lat

        h_tokens = H_lat_p // P
        w_tokens = W_lat_p // P

        timestep = torch.zeros(B, device=latent.device, dtype=latent.dtype)

        if self._backend in ("diffusers", "cosmos_predict2"):
            features = self._extract_features_hook(latent, timestep, h_tokens, w_tokens)
        else:
            _, intermediates = self.dit(
                latent, timestep=timestep, feature_layers=self._feature_layers,
            )
            feat_list = [intermediates[i] for i in self._feature_layers if i in intermediates]
            if not feat_list:
                final, _ = self.dit(latent, timestep=timestep)
                feat_list = [final]
            features = self.feature_projector(feat_list, h_tokens, w_tokens)

        if pad_h > 0 or pad_w > 0:
            features = features[:, :, :H_lat, :W_lat]

        return features

    def _extract_features_hook(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        h_tokens: int,
        w_tokens: int,
    ) -> torch.Tensor:
        """Extract intermediate features from diffusers/cosmos DiT via hooks."""
        collected: List[torch.Tensor] = []
        hooks = []

        # TODO: The block container attribute name may differ between
        # diffusers versions.  Common names: .transformer_blocks,
        # .blocks, .layers.  Adjust after inspection.
        block_container = None
        for attr in ("transformer_blocks", "blocks", "layers"):
            if hasattr(self.dit, attr):
                block_container = getattr(self.dit, attr)
                break

        if block_container is None:
            logger.warning(
                "Cannot find block container on DiT module (%s). "
                "Returning downsampled latent features.",
                type(self.dit).__name__,
            )
            fallback = rearrange(latent, "b c h w -> b (h w) c")
            return self.feature_projector([fallback] * len(self._feature_layers), h_tokens, w_tokens)

        def _make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                if output.dim() == 3:
                    collected.append(output.detach())
                else:
                    collected.append(rearrange(output.detach(), "b ... d -> b (...) d"))
            return hook_fn

        for idx in self._feature_layers:
            if idx < len(block_container):
                h = block_container[idx].register_forward_hook(_make_hook(idx))
                hooks.append(h)

        try:
            with torch.no_grad():
                # TODO: Forward signature for the diffusers transformer may
                # require hidden_states + timestep + encoder_hidden_states.
                # Using a null text conditioning tensor as placeholder.
                if latent.dim() == 4:
                    latent_5d = rearrange(latent, "b c h w -> b c 1 h w")
                else:
                    latent_5d = latent

                null_text = torch.zeros(
                    latent.shape[0], 1, self.cfg.hidden_dim,
                    device=latent.device, dtype=latent.dtype,
                )
                try:
                    self.dit(
                        hidden_states=latent_5d,
                        timestep=timestep,
                        encoder_hidden_states=null_text,
                    )
                except TypeError:
                    self.dit(latent_5d, timestep)
        finally:
            for h in hooks:
                h.remove()

        if not collected:
            fallback = rearrange(latent, "b c h w -> b (h w) c")
            collected = [fallback] * len(self._feature_layers)

        return self.feature_projector(collected, h_tokens, w_tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        semantic_ids: Optional[torch.Tensor] = None,
        point_prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode -> extract features -> heads.

        Args:
            x: Input image ``[B, C, H, W]``.
            semantic_ids: Optional per-pixel semantic class labels ``[B, H, W]``.
            point_prompts: Optional dict with interactive point annotations
                as produced by :func:`~neurons.utils.point_sampling.sample_point_prompts`.

        Returns:
            Dictionary with keys ``"semantic"``, ``"instance"``,
            ``"geometry"``, and optionally ``"semantic_ids"``.
        """
        original_dtype = x.dtype
        H_in, W_in = x.shape[-2], x.shape[-1]

        rgb = self.input_adapter(x)

        s = self.cfg.spatial_compression
        pad_h = (s - H_in % s) % s
        pad_w = (s - W_in % s) % s
        if pad_h > 0 or pad_w > 0:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="reflect")

        compute_dtype = self._dtype if self._backbone_loaded else original_dtype

        latent = self._encode_to_latent(rgb.to(dtype=compute_dtype))

        features = self._extract_features(latent)

        features = self.upsampler(features.to(dtype=original_dtype))

        if features.shape[-2] != H_in or features.shape[-1] != W_in:
            features = F.interpolate(features, size=(H_in, W_in), mode="bilinear", align_corners=False)

        if point_prompts is not None:
            features = features + self.point_encoder(
                pos_points=point_prompts["pos_points"],
                neg_points=point_prompts["neg_points"],
                target_semantic_ids=point_prompts["target_semantic_ids"],
                target_instance_ids=point_prompts["target_instance_ids"],
                spatial_shape=features.shape[2:],
            )

        out: Dict[str, torch.Tensor] = {
            "semantic": self.head_semantic(features),
            "instance": self.head_instance(features),
            "geometry": self.head_geometry(features),
        }
        if semantic_ids is not None:
            out["semantic_ids"] = semantic_ids
        return out

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze the DiT backbone (VAE encoder is always frozen)."""
        self.dit.requires_grad_(False)
        logger.debug("DiT backbone frozen.")

    def unfreeze_encoder(self) -> None:
        """Unfreeze the DiT backbone for fine-tuning."""
        self.dit.requires_grad_(True)
        logger.debug("DiT backbone unfrozen.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_output_channels(self) -> int:
        return self.num_classes


# ---------------------------------------------------------------------------
# Fit verification
# ---------------------------------------------------------------------------

def verify_fit(
    variant: str = "2B",
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    num_classes: int = 16,
    emb_dim: int = 16,
    feature_size: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Verify whether a Cosmos-Predict2.5 variant fits the 3-stage task.

    Checks input/output dimensionality, sequence length constraints,
    estimated memory footprint, and tokenizer spatial requirements.

    Args:
        variant: ``"2B"`` or ``"14B"``.
        input_shape: ``(B, C, H, W)`` test shape.
        num_classes: Semantic classes.
        emb_dim: Instance embedding dim.
        feature_size: Intermediate feature channels.
        device: Target device for memory estimation.

    Returns:
        Dict with ``compatible`` bool, ``warnings`` list, ``errors`` list,
        and detailed ``checks`` dict.
    """
    variant = variant.upper()
    if variant not in _VARIANT_CONFIGS:
        return {"compatible": False, "errors": [f"Unknown variant: {variant}"], "warnings": [], "checks": {}}

    cfg = _VARIANT_CONFIGS[variant]
    B, C, H, W = input_shape
    results: Dict[str, Any] = {
        "variant": variant,
        "compatible": True,
        "warnings": [],
        "errors": [],
        "checks": {},
    }

    s = cfg.spatial_compression
    H_lat, W_lat = H // s, W // s
    results["checks"]["latent_spatial"] = (H_lat, W_lat)

    if H_lat < 1 or W_lat < 1:
        results["errors"].append(
            f"Input {H}x{W} too small for {s}x spatial compression. "
            f"Minimum input: {s}x{s}."
        )
        results["compatible"] = False

    if H % s != 0 or W % s != 0:
        results["warnings"].append(
            f"Input {H}x{W} not divisible by spatial_compression={s}. "
            f"Padding will be applied."
        )

    P = cfg.patch_size
    H_lat_padded = H_lat + (P - H_lat % P) % P
    W_lat_padded = W_lat + (P - W_lat % P) % P
    seq_len = (H_lat_padded // P) * (W_lat_padded // P)
    results["checks"]["sequence_length"] = seq_len
    results["checks"]["max_sequence_length"] = cfg.max_sequence_length

    if seq_len > cfg.max_sequence_length:
        results["errors"].append(
            f"Sequence length {seq_len} exceeds model maximum "
            f"{cfg.max_sequence_length}. Reduce input resolution."
        )
        results["compatible"] = False

    S = _SPATIAL_DIMS
    geom_ch = S + S * S + 4
    results["checks"]["output_channels"] = {
        "semantic": num_classes,
        "instance": emb_dim,
        "geometry": geom_ch,
        "total": num_classes + emb_dim + geom_ch,
    }

    param_bytes = cfg.hidden_dim * cfg.num_layers * cfg.hidden_dim * 4 * 2
    backbone_gb = param_bytes / (1024 ** 3)
    head_params = (feature_size * 64 + 64 * (num_classes + emb_dim + geom_ch)) * 9
    head_gb = head_params * 4 / (1024 ** 3)
    activation_gb = B * seq_len * cfg.hidden_dim * 4 / (1024 ** 3)
    total_gb = backbone_gb + head_gb + activation_gb

    results["checks"]["memory_estimate_gb"] = {
        "backbone_params": round(backbone_gb, 2),
        "head_params": round(head_gb, 4),
        "activations": round(activation_gb, 2),
        "total": round(total_gb, 2),
        "variant_vram_recommended": cfg.estimated_vram_gb,
    }

    if device == "cuda" and torch.cuda.is_available():
        available_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        results["checks"]["memory_estimate_gb"]["available"] = round(available_gb, 2)
        if total_gb > available_gb * 0.85:
            results["warnings"].append(
                f"Estimated {total_gb:.1f} GB may exceed 85% of available "
                f"{available_gb:.1f} GB GPU memory."
            )
    else:
        results["warnings"].append("CUDA not available -- cannot check GPU memory.")

    if variant == "14B":
        results["warnings"].append(
            "14B variant typically requires multi-GPU or CPU offloading. "
            "Consider gradient checkpointing and mixed precision."
        )

    results["checks"]["stage_compatibility"] = {
        "stage1_semantic": "Compatible -- standard pixel classification via logit head.",
        "stage2_instance": "Compatible -- embedding head produces per-pixel vectors.",
        "stage3_geometry": (
            "Compatible with caveats -- Cosmos backbone is trained on natural "
            "images/video, not EM data. Direction and covariance heads are "
            "randomly initialized and require fine-tuning."
        ),
    }

    if not results["errors"]:
        logger.info("verify_fit(%s): PASS -- %d warnings", variant, len(results["warnings"]))
    else:
        results["compatible"] = False
        logger.warning("verify_fit(%s): FAIL -- %s", variant, results["errors"])

    return results
