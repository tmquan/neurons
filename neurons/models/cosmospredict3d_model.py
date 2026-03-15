"""
Cosmos-Predict2.5 **3D** model wrapper for volumetric connectomics segmentation.

Adapts the Cosmos-Predict2.5 DiT backbone (2B or 14B) as a feature
extractor for the three-stage volumetric segmentation task:

- **Semantic**: per-voxel class logits  (``num_classes`` channels)
- **Instance**: per-voxel embedding vectors  (``emb_dim`` channels)
- **Geometry**: per-voxel direction, covariance, and RGBA reconstruction

Cosmos-Predict2.5 is natively a video model with temporal + spatial
dimensions.  For volumetric EM data the depth axis maps directly to the
temporal axis, making the 3D adaptation architecturally natural:

    EM volume  [B, C, D, H, W]  <->  video  [B, C, T, H, W]

The VAE encoder compresses along all three axes (temporal_compression x
for depth, spatial_compression x for height/width).  The DiT backbone
then processes the full 3D latent grid.

References:
    - https://github.com/nvidia-cosmos/cosmos-predict2.5
    - HuggingFace: nvidia/Cosmos-Predict2.5-2B, nvidia/Cosmos-Predict2.5-14B
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from neurons.models.point_prompt_encoder import PointPromptEncoder

logger = logging.getLogger(__name__)

_SPATIAL_DIMS = 3
_CONV = nn.Conv3d


def _NORM(ch: int) -> nn.GroupNorm:
    num_groups = max(g for g in (1, 2, 4, 8, 16, 32) if ch % g == 0)
    return nn.GroupNorm(num_groups, ch)


class _PointwiseLinear(nn.Module):
    """Drop-in replacement for Conv{2,3}d(k=1) using nn.Linear.

    Avoids non-contiguous gradient strides that cause DDP warnings.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = rearrange(x, "b c ... -> b ... c").to(self.linear.weight.dtype)
        return rearrange(self.linear(x_in), "b ... c -> b c ...")


# ---------------------------------------------------------------------------
# Variant configuration  (shared with the 2D model)
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
        temporal_compression=4,
        estimated_vram_gb=12.0,
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
        temporal_compression=4,
        estimated_vram_gb=48.0,
        max_sequence_length=32768,
    ),
}


# ---------------------------------------------------------------------------
# HuggingFace weight download
# ---------------------------------------------------------------------------

def _download_from_hf(
    repo_id: str,
    revision: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Download model snapshot from HuggingFace Hub.

    In DDP training, rank 0 downloads first while other ranks wait at a
    barrier, then all ranks resolve the cached path without re-downloading.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for Cosmos-Predict2.5 weight "
            "download.  Install with: pip install huggingface_hub"
        )

    import torch.distributed as dist

    cache_dir = cache_dir or str(
        Path.home() / ".cache" / "neurons" / "cosmos_predict25"
    )
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if rank == 0:
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                ignore_patterns=["*.md", "*.txt", "examples/*", "docs/*"],
            )
            logger.info("Downloaded %s (rev=%s) -> %s", repo_id, revision, local_path)
        except Exception as exc:
            logger.warning(
                "HuggingFace download failed for %s (rev=%s): %s.  "
                "Falling back to random initialisation.",
                repo_id, revision, exc,
            )
            if is_distributed:
                dist.barrier()
            raise

    if is_distributed:
        dist.barrier()

    if rank != 0:
        local_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=True,
            ignore_patterns=["*.md", "*.txt", "examples/*", "docs/*"],
        )
        logger.info("Downloaded %s (rev=%s) -> %s", repo_id, revision, local_path)

    return Path(local_path)


# ---------------------------------------------------------------------------
# 3-D input channel adapter
# ---------------------------------------------------------------------------

def _adapt_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Adapt input channels to 3-ch RGB expected by Cosmos.

    For single-channel EM volumes, repeats grayscale to 3 channels.
    This preserves the VAE encoder's pretrained input distribution
    without introducing learnable parameters.
    """
    if x.shape[1] == 3:
        return x
    return repeat(x, "b 1 ... -> b 3 ...")


# ---------------------------------------------------------------------------
# Lightweight 3-D DiT block (standalone fallback)
# ---------------------------------------------------------------------------

class _DiTBlock(nn.Module):
    """Single DiT block with adaptive layer norm (standalone fallback)."""

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
            mod = self.adaLN_modulation(timestep_emb)
            shift1, scale1, shift2, scale2 = rearrange(
                mod, "b (n d) -> n b d", n=4,
            ).unbind(0)
            h = (
                self.norm1(x)
                * (1 + rearrange(scale1, "b d -> b 1 d"))
                + rearrange(shift1, "b d -> b 1 d")
            )
        else:
            h = self.norm1(x)

        h, _ = self.attn(h, h, h)
        x = x + h

        if timestep_emb is not None:
            h = (
                self.norm2(x)
                * (1 + rearrange(scale2, "b d -> b 1 d"))
                + rearrange(shift2, "b d -> b 1 d")
            )
        else:
            h = self.norm2(x)

        return x + self.mlp(h)


class _StandaloneDiT3D(nn.Module):
    """Minimal 3-D DiT matching Cosmos-Predict2.5 shape.

    Patch embedding operates on volumetric patches
    ``(P_d, P_h, P_w) = (patch_size,) * 3`` producing a 1-D sequence of
    tokens processed by self-attention blocks.
    """

    def __init__(self, cfg: _VariantConfig) -> None:
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.patch_size = cfg.patch_size
        self.latent_channels = cfg.latent_channels

        patch_input_dim = cfg.latent_channels * cfg.patch_size ** 3
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
        """Run 3-D DiT and optionally return intermediate features.

        Args:
            latent: ``[B, C_lat, D_lat, H_lat, W_lat]``.
            timestep: Scalar or ``[B]`` for adaptive norm.
            feature_layers: Block indices whose outputs to collect.

        Returns:
            ``(final_hidden [B, N, D], intermediates {idx: [B, N, D]})``.
        """
        _param_dtype = self.patch_embed.weight.dtype
        latent = latent.to(dtype=_param_dtype)

        B, _C, _D, _H, _W = latent.shape
        P = self.patch_size

        patches = rearrange(
            latent,
            "b c (d p1) (h p2) (w p3) -> b (d h w) (c p1 p2 p3)",
            p1=P, p2=P, p3=P,
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
# 3-D feature pyramid + upsampler
# ---------------------------------------------------------------------------

class _FeatureProjector3D(nn.Module):
    """Fuse multi-layer DiT features into a 3-D spatial feature map."""

    def __init__(
        self,
        hidden_dim: int,
        num_feature_layers: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        total_in = hidden_dim * num_feature_layers
        self.proj = nn.Sequential(
            _PointwiseLinear(total_in, out_dim * 2),
            _NORM(out_dim * 2),
            nn.GELU(),
            _PointwiseLinear(out_dim * 2, out_dim),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        d: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        spatial = [
            rearrange(f, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
            for f in features
        ]
        fused = torch.cat(spatial, dim=1)
        return self.proj(fused)


class _ProgressiveUpsampler3D(nn.Module):
    """Progressive 3-D upsampling (each stage doubles spatial dims)."""

    def __init__(self, in_dim: int, out_dim: int, num_stages: int) -> None:
        super().__init__()
        dims = self._interpolate_dims(in_dim, out_dim, num_stages + 1)
        layers: List[nn.Module] = []
        for i in range(num_stages):
            layers.append(nn.Sequential(
                nn.ConvTranspose3d(
                    dims[i], dims[i + 1],
                    kernel_size=4, stride=2, padding=1,
                ),
                _NORM(dims[i + 1]),
                nn.GELU(),
            ))
        self.stages = nn.ModuleList(layers)

    @staticmethod
    def _interpolate_dims(start: int, end: int, n: int) -> List[int]:
        if n <= 1:
            return [start]
        step = (end - start) / (n - 1)
        dims = [
            max(8, int(round((start + i * step) / 8)) * 8)
            for i in range(n)
        ]
        dims[0], dims[-1] = start, end
        return dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class _DecoderAdapter3D(nn.Module):
    """Reuses pretrained VAE decoder for multi-head volumetric segmentation.

    Replaces the decoder's final output convolution with three parallel
    task heads while preserving all pretrained upsampling weights.

    Freeze policy:
      - Decoder body (early / mid blocks): frozen
      - Last up-block + output norm: trainable
      - Task heads: trainable (randomly initialised)
    """

    def __init__(
        self,
        vae_decoder: Optional[nn.Module],
        latent_channels: int,
        feature_size: int,
        num_classes: int,
        emb_dim: int,
        geom_channels: int,
        spatial_compression: int,
        temporal_compression: int,
        dropout: float = 0.0,
        freeze_vae_decoder: bool = False,
    ) -> None:
        super().__init__()
        self._has_pretrained = vae_decoder is not None

        if vae_decoder is not None:
            self.to_latent = _PointwiseLinear(feature_size, latent_channels)
            self.decoder_body = vae_decoder
            self._hidden_ch = self._replace_conv_out()
            if freeze_vae_decoder:
                self._freeze_body()
        else:
            self.to_latent = None
            num_up_spatial = int(math.log2(spatial_compression))
            num_up_temporal = int(math.log2(temporal_compression))
            num_stages = max(num_up_spatial, num_up_temporal)
            self.decoder_body = _ProgressiveUpsampler3D(
                in_dim=feature_size, out_dim=feature_size,
                num_stages=num_stages,
            )
            self._hidden_ch = feature_size

        self.head_semantic = nn.Sequential(
            _CONV(self._hidden_ch, 64, 3, padding=1), _NORM(64),
            nn.ReLU(inplace=True), nn.Dropout3d(dropout),
            _CONV(64, num_classes, 1),
        )
        self.head_instance = nn.Sequential(
            _CONV(self._hidden_ch, 64, 3, padding=1), _NORM(64),
            nn.ReLU(inplace=True), nn.Dropout3d(dropout),
            _CONV(64, emb_dim, 1),
        )
        self.head_geometry = nn.Sequential(
            _CONV(self._hidden_ch, 64, 3, padding=1), _NORM(64),
            nn.ReLU(inplace=True), nn.Dropout3d(dropout),
            _CONV(64, geom_channels, 1),
        )

    def _replace_conv_out(self) -> int:
        for attr in ("conv_out", "output_conv", "proj_out", "final_conv"):
            if hasattr(self.decoder_body, attr):
                final = getattr(self.decoder_body, attr)
                if hasattr(final, "in_channels"):
                    ch = final.in_channels
                elif hasattr(final, "weight") and final.weight.dim() >= 2:
                    ch = final.weight.shape[1]
                else:
                    continue
                setattr(self.decoder_body, attr, nn.Identity())
                logger.info(
                    "Replaced decoder.%s (hidden_ch=%d) with Identity.", attr, ch,
                )
                return ch
        logger.warning(
            "Could not find decoder final conv; using latent_channels as hidden_ch."
        )
        return self.to_latent.linear.out_features

    def _freeze_body(self) -> None:
        for p in self.decoder_body.parameters():
            p.requires_grad = False
        for attr in ("up_blocks", "up"):
            if hasattr(self.decoder_body, attr):
                blocks = getattr(self.decoder_body, attr)
                if hasattr(blocks, "__len__") and len(blocks) > 0:
                    for p in blocks[-1].parameters():
                        p.requires_grad = True
                break
        for attr in ("conv_norm_out", "norm_out"):
            if hasattr(self.decoder_body, attr):
                for p in getattr(self.decoder_body, attr).parameters():
                    p.requires_grad = True
                break

    def _unfreeze_body(self) -> None:
        for p in self.decoder_body.parameters():
            p.requires_grad = True

    def forward(
        self, features: torch.Tensor, target_size: tuple,
    ) -> Dict[str, torch.Tensor]:
        if self._has_pretrained:
            latent = self.to_latent(features)
            body_dtype = next(self.decoder_body.parameters()).dtype
            decoded = self.decoder_body(latent.to(body_dtype))
            if isinstance(decoded, (tuple, list)):
                decoded = decoded[0]
            if hasattr(decoded, "sample"):
                decoded = decoded.sample
            decoded = decoded.to(features.dtype)
        else:
            decoded = self.decoder_body(features)
        if decoded.shape[-3:] != target_size:
            decoded = F.interpolate(
                decoded, size=target_size, mode="trilinear", align_corners=False,
            )
        return {
            "semantic": self.head_semantic(decoded),
            "instance": self.head_instance(decoded),
            "geometry": self.head_geometry(decoded),
        }


# ---------------------------------------------------------------------------
# Main 3D model
# ---------------------------------------------------------------------------

class CosmosPredict3DWrapper(nn.Module):
    """Cosmos-Predict2.5 adapted for **volumetric** connectomics segmentation.

    Three parallel output heads produce:

    - ``semantic``  [B, num_classes, D, H, W]
    - ``instance``  [B, emb_dim, D, H, W]
    - ``geometry``  [B, G, D, H, W]  where G = 3 + 9 + 4 = 16

    Because Cosmos-Predict2.5 is natively a video model, the depth axis
    of the EM volume is mapped to the temporal axis of the backbone,
    making the 3-D adaptation architecturally natural.

    Args:
        in_channels: Number of input channels (1 for EM volumes).
        num_classes: Semantic segmentation classes.
        emb_dim: Instance embedding dimensionality.
        feature_size: Internal feature map channel count after projection.
        variant: ``"2B"`` or ``"14B"`` model variant.
        checkpoint_variant: HuggingFace revision string.
        dtype: Weight dtype (``"bf16"``, ``"fp16"``, ``"fp32"``).
        freeze_dit_backbone: Whether to freeze the pretrained DiT backbone.
        feature_layers: DiT block indices to extract features from.
        cache_dir: HuggingFace download cache directory.
        hf_token: HuggingFace authentication token.
        dropout: Dropout probability for heads.

    Example::

        >>> model = CosmosPredict3DWrapper(
        ...     in_channels=1, num_classes=16, variant="2B",
        ... )
        >>> x = torch.randn(1, 1, 32, 64, 64)
        >>> out = model(x)
        >>> out["semantic"].shape   # [1, 16, 32, 64, 64]
        >>> out["instance"].shape   # [1, 16, 32, 64, 64]
        >>> out["geometry"].shape   # [1, 16, 32, 64, 64]
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
        freeze_dit_backbone: bool = False,
        freeze_vae_decoder: bool = False,
        freeze_vae_encoder: bool = True,
        gradient_checkpointing: bool = False,
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
                f"Unknown variant '{variant}'.  "
                f"Choose from: {list(_VARIANT_CONFIGS)}"
            )

        self.variant = variant
        self.cfg = _VARIANT_CONFIGS[variant]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.feature_size = feature_size
        self.spatial_dims = _SPATIAL_DIMS
        self.dropout = dropout

        S = _SPATIAL_DIMS
        self.geom_channels = S + S * S + 4

        self._dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        self._freeze_dit_backbone = freeze_dit_backbone
        self._freeze_vae_decoder = freeze_vae_decoder
        self._freeze_vae_encoder = freeze_vae_encoder
        self._gradient_checkpointing = gradient_checkpointing

        if feature_layers is not None:
            self._feature_layers = sorted(feature_layers)
        else:
            n = self.cfg.num_layers
            self._feature_layers = sorted(
                {n // 4, n // 2, 3 * n // 4, n - 1}
            )

        s = self.cfg.spatial_compression
        t = self.cfg.temporal_compression
        lc = self.cfg.latent_channels
        self._fallback_down = nn.Sequential(
            nn.Conv3d(3, lc * 2, kernel_size=(t, s, s), stride=(t, s, s)),
            _NORM(lc * 2),
            nn.GELU(),
            nn.Conv3d(lc * 2, lc, kernel_size=1),
        )

        self._backbone_loaded = False
        self._build_backbone(cache_dir, hf_token, checkpoint_variant)

        self.feature_projector = _FeatureProjector3D(
            hidden_dim=self.cfg.hidden_dim,
            num_feature_layers=len(self._feature_layers),
            out_dim=feature_size,
        ).float()

        if self._backend in ("diffusers", "cosmos_predict2"):
            self._register_persistent_hooks()

        self.decoder_adapter = _DecoderAdapter3D(
            vae_decoder=self.vae_decoder,
            latent_channels=self.cfg.latent_channels,
            feature_size=feature_size,
            num_classes=num_classes,
            emb_dim=emb_dim,
            geom_channels=self.geom_channels,
            spatial_compression=self.cfg.spatial_compression,
            temporal_compression=self.cfg.temporal_compression,
            dropout=dropout,
            freeze_vae_decoder=freeze_vae_decoder,
        )

        self.point_encoder = PointPromptEncoder(
            num_classes=num_classes,
            feature_size=feature_size,
            spatial_dims=_SPATIAL_DIMS,
        )

        if self.vae_encoder is not None and freeze_vae_encoder:
            self.vae_encoder.requires_grad_(False)
            self.vae_encoder.eval()

        if freeze_dit_backbone:
            self.freeze_dit_backbone()
        else:
            self.dit.train()

        self._make_params_contiguous()

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

        logger.info(
            "CosmosPredict3DWrapper initialised: variant=%s, "
            "feature_layers=%s, backbone_loaded=%s, frozen=%s, "
            "grad_ckpt=%s, params=%s (trainable=%s)",
            variant, self._feature_layers, self._backbone_loaded,
            freeze_dit_backbone, self._gradient_checkpointing,
            f"{self.get_num_parameters(trainable_only=False):,}",
            f"{self.get_num_parameters(trainable_only=True):,}",
        )

    def _apply(self, fn):
        """Extend device/dtype placement to the untracked full-VAE reference.

        ``_vae_ref`` is a plain Python list (not ``nn.ModuleList``) to avoid
        double-registering encoder/decoder parameters in ``state_dict()``.
        This override ensures that auxiliary VAE components (e.g. quant_conv)
        are moved together with the rest of the model.
        """
        super()._apply(fn)
        if hasattr(self, "_vae_ref") and self._vae_ref:
            self._vae_ref[0]._apply(fn)
        return self

    def _make_params_contiguous(self) -> None:
        """Ensure all parameter data tensors are contiguous for DDP."""
        for p in self.parameters():
            if not p.data.is_contiguous():
                p.data = p.data.contiguous()

    # ------------------------------------------------------------------
    # Backbone construction
    # ------------------------------------------------------------------

    def _build_backbone(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> None:
        self.vae_encoder: Optional[nn.Module] = None
        self.vae_decoder: Optional[nn.Module] = None
        self.dit: nn.Module

        _saved_dtype = torch.get_default_dtype()
        try:
            loaded = (
                self._try_load_diffusers(cache_dir, hf_token, checkpoint_variant)
                or self._try_load_cosmos_package(
                    cache_dir, hf_token, checkpoint_variant,
                )
                or self._try_load_raw_checkpoint(
                    cache_dir, hf_token, checkpoint_variant,
                )
            )
        finally:
            torch.set_default_dtype(_saved_dtype)

        if not loaded:
            logger.warning(
                "No pretrained weights loaded -- using randomly initialised "
                "3-D DiT backbone (%s architecture).",
                self.variant,
            )
            self._build_standalone_backbone()

    def _try_load_diffusers(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        try:
            from diffusers import (  # type: ignore[import-untyped]
                CosmosTransformer3DModel,
            )
            from diffusers import AutoencoderKLWan as _VAEClass  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("diffusers Cosmos classes not available.")
            return False

        try:
            local_path = _download_from_hf(
                self.cfg.hf_repo_id,
                revision=self.cfg.hf_revision,
                cache_dir=cache_dir,
                token=hf_token,
            )
        except Exception as exc:
            logger.warning("HuggingFace download failed: %s", exc)
            return False

        try:
            transformer = CosmosTransformer3DModel.from_pretrained(
                str(local_path),
                subfolder="transformer",
                torch_dtype=self._dtype,
            )
            vae = _VAEClass.from_pretrained(
                str(local_path),
                subfolder="vae",
                torch_dtype=self._dtype,
            )

            vae = vae.to(self._dtype)
            self._vae_ref = [vae]
            self.vae_encoder = vae.encoder
            self.vae_decoder = vae.decoder

            self.dit = transformer.to(self._dtype)
            self._backbone_loaded = True
            self._backend = "diffusers"
            logger.info(
                "Loaded 3-D backbone + VAE via diffusers (local snapshot).",
            )
            return True
        except Exception as exc:
            logger.warning("diffusers load failed: %s", exc)
            return False

    def _try_load_cosmos_package(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        try:
            # TODO: Exact import depends on installed cosmos_predict2 version.
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
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "encoder"):
                self.vae_encoder = pipe.vae.encoder.to(self._dtype)
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "decoder"):
                self.vae_decoder = pipe.vae.decoder.to(self._dtype)

            if hasattr(pipe, "dit"):
                self.dit = pipe.dit.to(self._dtype)
            elif hasattr(pipe, "transformer"):
                self.dit = pipe.transformer.to(self._dtype)
            else:
                logger.warning(
                    "Could not locate DiT module on cosmos_predict2 pipeline."
                )
                return False

            self._backbone_loaded = True
            self._backend = "cosmos_predict2"
            logger.info("Loaded 3-D backbone via cosmos_predict2 package.")
            return True
        except Exception as exc:
            logger.warning("cosmos_predict2 load failed: %s", exc)
            return False

    def _try_load_raw_checkpoint(
        self,
        cache_dir: Optional[str],
        hf_token: Optional[str],
        checkpoint_variant: str,
    ) -> bool:
        try:
            local_path = _download_from_hf(
                self.cfg.hf_repo_id,
                revision=self.cfg.hf_revision,
                cache_dir=cache_dir,
                token=hf_token,
            )
        except Exception:
            return False

        local_path = Path(local_path)

        # --- Load VAE from snapshot via diffusers ---
        try:
            from diffusers import AutoencoderKLWan  # type: ignore[import-untyped]

            vae = AutoencoderKLWan.from_pretrained(
                str(local_path), subfolder="vae",
                torch_dtype=self._dtype,
            )
            vae = vae.to(self._dtype)
            self._vae_ref = [vae]
            self.vae_encoder = vae.encoder
            self.vae_decoder = vae.decoder
            logger.info("Loaded VAE encoder + decoder from snapshot.")
        except Exception as exc:
            logger.warning("Could not load VAE from snapshot: %s", exc)

        # --- Load DiT: try diffusers first, then raw weights ---
        try:
            from diffusers import CosmosTransformer3DModel  # type: ignore[import-untyped]

            self.dit = CosmosTransformer3DModel.from_pretrained(
                str(local_path), subfolder="transformer",
                torch_dtype=self._dtype,
            ).to(self._dtype)
            self._backbone_loaded = True
            self._backend = "diffusers"
            logger.info("Loaded DiT transformer from snapshot via diffusers.")
            return True
        except Exception as exc:
            logger.warning("diffusers DiT load from snapshot failed: %s", exc)

        self._build_standalone_backbone()

        transformer_dir = local_path / "transformer"
        ckpt_files = (
            list(transformer_dir.glob("*.safetensors"))
            + list(transformer_dir.glob("*.pt"))
        ) if transformer_dir.is_dir() else (
            list(local_path.glob("**/*.safetensors"))
            + list(local_path.glob("**/*.pt"))
        )
        if not ckpt_files:
            logger.warning("No checkpoint files found in %s.", local_path)
            return False

        loaded_any = False
        for ckpt_file in ckpt_files:
            try:
                if ckpt_file.suffix == ".safetensors":
                    from safetensors.torch import load_file  # type: ignore[import-untyped]
                    state = load_file(str(ckpt_file), device="cpu")
                else:
                    state = torch.load(
                        str(ckpt_file), map_location="cpu", weights_only=True,
                    )
                missing, unexpected = self.dit.load_state_dict(
                    state, strict=False,
                )
                if missing:
                    logger.debug(
                        "Missing keys from %s: %d", ckpt_file.name, len(missing),
                    )
                if unexpected:
                    logger.debug(
                        "Unexpected keys in %s: %d",
                        ckpt_file.name, len(unexpected),
                    )
                loaded_any = True
            except Exception as exc:
                logger.warning("Failed to load %s: %s", ckpt_file.name, exc)

        if loaded_any:
            self._backbone_loaded = True
            self._backend = "raw_checkpoint"
            logger.info(
                "Loaded weights from raw checkpoint into standalone 3-D DiT."
            )
        return loaded_any

    def _build_standalone_backbone(self) -> None:
        self.dit = _StandaloneDiT3D(self.cfg)
        self._backend = "standalone"

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space volume ``[B, 3, D, H, W]`` to latent grid."""
        if hasattr(self, "_vae_ref") and self._vae_ref:
            vae = self._vae_ref[0]
            ctx = torch.no_grad() if self._freeze_vae_encoder else torch.enable_grad()
            with ctx:
                enc = vae.encode(x)
                if hasattr(enc, "latent_dist"):
                    latent = enc.latent_dist.mode()
                elif hasattr(enc, "sample"):
                    latent = enc.sample
                else:
                    latent = enc
                return latent.to(dtype=x.dtype)

        if self.vae_encoder is not None:
            ctx = torch.no_grad() if self._freeze_vae_encoder else torch.enable_grad()
            with ctx:
                enc = self.vae_encoder(x)
                if hasattr(enc, "latent_dist"):
                    latent = enc.latent_dist.mode()
                elif hasattr(enc, "sample"):
                    latent = enc.sample
                else:
                    latent = enc
                return latent.to(dtype=x.dtype)

        return self._conv_downsample(x)

    def _conv_downsample(self, x: torch.Tensor) -> torch.Tensor:
        return self._fallback_down.to(device=x.device, dtype=x.dtype)(x)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, latent: torch.Tensor) -> torch.Tensor:
        """Run 3-D DiT backbone and extract multi-layer features.

        Returns ``[B, feature_size, D_lat, H_lat, W_lat]``.
        """
        B, _C, D_lat, H_lat, W_lat = latent.shape

        dit_cfg = getattr(self.dit, "config", None)
        dit_ps = getattr(dit_cfg, "patch_size", None)
        if isinstance(dit_ps, (list, tuple)) and len(dit_ps) == 3:
            p_t, p_h, p_w = dit_ps
        else:
            p_t = p_h = p_w = self.cfg.patch_size

        pad_d = (p_t - D_lat % p_t) % p_t
        pad_h = (p_h - H_lat % p_h) % p_h
        pad_w = (p_w - W_lat % p_w) % p_w
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            latent = F.pad(
                latent, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate",
            )
        D_p = D_lat + pad_d
        H_p = H_lat + pad_h
        W_p = W_lat + pad_w

        d_tok, h_tok, w_tok = D_p // p_t, H_p // p_h, W_p // p_w

        timestep = torch.zeros(B, device=latent.device, dtype=latent.dtype)

        if self._backend in ("diffusers", "cosmos_predict2"):
            features = self._extract_features_hook(
                latent, timestep, d_tok, h_tok, w_tok,
            )
        else:
            final, intermediates = self.dit(
                latent, timestep=timestep,
                feature_layers=self._feature_layers,
            )
            feat_list = [
                intermediates[i]
                for i in self._feature_layers
                if i in intermediates
            ]
            if not feat_list:
                feat_list = [final]
            feat_list = [f.float() for f in feat_list]
            features = self.feature_projector(
                feat_list, d_tok, h_tok, w_tok,
            )

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            features = features[:, :, :D_lat, :H_lat, :W_lat]

        return features

    def _register_persistent_hooks(self) -> None:
        """Register forward hooks on DiT blocks once (called from __init__)."""
        self._hook_buffer: List[torch.Tensor] = []
        self._hook_handles: List[Any] = []
        self._hook_block_container = None
        self._hooks_active = False

        for attr in ("transformer_blocks", "blocks", "layers"):
            if hasattr(self.dit, attr):
                self._hook_block_container = getattr(self.dit, attr)
                break

        if self._hook_block_container is None:
            return

        def _make_hook(_idx: int):
            def hook_fn(_module: nn.Module, _input: Any, output: Any) -> None:
                if not self._hooks_active:
                    return
                out = output[0] if isinstance(output, tuple) else output
                if self._freeze_dit_backbone:
                    out = out.detach()
                if out.dim() == 3:
                    self._hook_buffer.append(out)
                else:
                    self._hook_buffer.append(rearrange(out, "b ... d -> b (...) d"))
            return hook_fn

        for idx in self._feature_layers:
            if idx < len(self._hook_block_container):
                h = self._hook_block_container[idx].register_forward_hook(
                    _make_hook(idx),
                )
                self._hook_handles.append(h)

    def _extract_features_hook(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        d_tok: int,
        h_tok: int,
        w_tok: int,
    ) -> torch.Tensor:
        """Extract intermediate features from diffusers / cosmos DiT."""
        if self._hook_block_container is None:
            logger.warning(
                "Cannot find block container on DiT (%s).  "
                "Returning conv-downsampled latent features.",
                type(self.dit).__name__,
            )
            fallback = rearrange(latent, "b c d h w -> b (d h w) c").float()
            return self.feature_projector(
                [fallback] * len(self._feature_layers),
                d_tok, h_tok, w_tok,
            )

        self._hook_buffer.clear()
        self._hooks_active = True

        try:
            ctx = torch.no_grad() if self._freeze_dit_backbone else torch.enable_grad()
            with ctx:
                B = latent.shape[0]
                dit_cfg = getattr(self.dit, "config", None)
                text_dim = getattr(dit_cfg, "crossattn_proj_in_channels", 1024)
                null_text = torch.zeros(B, 1, text_dim, device=latent.device, dtype=latent.dtype)

                img_dim_in = getattr(dit_cfg, "img_context_dim_in", None)
                img_tokens = getattr(dit_cfg, "img_context_num_tokens", 256)
                if img_dim_in:
                    null_img = torch.zeros(B, img_tokens, img_dim_in, device=latent.device, dtype=latent.dtype)
                    enc_hidden = (null_text, null_img)
                else:
                    enc_hidden = null_text

                padding_mask = torch.ones(1, 1, latent.shape[-2], latent.shape[-1], device=latent.device, dtype=latent.dtype)
                null_condition = torch.zeros(B, 1, *latent.shape[2:], device=latent.device, dtype=latent.dtype)

                self.dit(
                    hidden_states=latent,
                    timestep=timestep,
                    encoder_hidden_states=enc_hidden,
                    condition_mask=null_condition,
                    padding_mask=padding_mask,
                )
        finally:
            self._hooks_active = False

        collected = list(self._hook_buffer)
        self._hook_buffer.clear()

        expected = len(self._feature_layers)
        if len(collected) < expected:
            fallback = rearrange(latent, "b c d h w -> b (d h w) c")
            while len(collected) < expected:
                collected.append(fallback)

        collected = [f.float() for f in collected]
        return self.feature_projector(collected, d_tok, h_tok, w_tok)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        semantic_ids: Optional[torch.Tensor] = None,
        point_prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode -> DiT features -> 3-D heads.

        Args:
            x: Input volume ``[B, C, D, H, W]``.
            semantic_ids: Optional per-voxel class labels ``[B, D, H, W]``.
            point_prompts: Optional interactive point annotations.

        Returns:
            Dict with ``"semantic"``, ``"instance"``, ``"geometry"``
            and optionally ``"semantic_ids"``.
        """
        original_dtype = x.dtype
        D_in, H_in, W_in = x.shape[-3], x.shape[-2], x.shape[-1]

        rgb = _adapt_to_rgb(x)

        s = self.cfg.spatial_compression
        t = self.cfg.temporal_compression
        pad_d = (t - D_in % t) % t
        pad_h = (s - H_in % s) % s
        pad_w = (s - W_in % s) % s
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")

        compute_dtype = self._dtype if self._backbone_loaded else original_dtype
        latent = self._encode_to_latent(rgb.to(dtype=compute_dtype))

        features = self._extract_features(latent)

        features = features.to(dtype=original_dtype)

        if point_prompts is not None:
            features = features + self.point_encoder(
                pos_points=point_prompts["pos_points"],
                neg_points=point_prompts["neg_points"],
                target_semantic_ids=point_prompts["target_semantic_ids"],
                target_instance_ids=point_prompts["target_instance_ids"],
                spatial_shape=features.shape[2:],
            )

        out = self.decoder_adapter(features, target_size=(D_in, H_in, W_in))
        if semantic_ids is not None:
            out["semantic_ids"] = semantic_ids
        return out

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_dit_backbone(self) -> None:
        self.dit.requires_grad_(False)
        self._freeze_dit_backbone = True
        logger.info("DiT backbone frozen (%s trainable params).",
                     f"{self.get_num_parameters(True):,}")

    def unfreeze_dit_backbone(self) -> None:
        self.dit.requires_grad_(True)
        self._freeze_dit_backbone = False
        logger.info("DiT backbone unfrozen (%s trainable params).",
                     f"{self.get_num_parameters(True):,}")

    def freeze_vae_encoder(self) -> None:
        if self.vae_encoder is not None:
            self.vae_encoder.requires_grad_(False)
            self.vae_encoder.eval()
            self._freeze_vae_encoder = True
            logger.info("VAE encoder frozen.")

    def unfreeze_vae_encoder(self) -> None:
        if self.vae_encoder is not None:
            self.vae_encoder.requires_grad_(True)
            self.vae_encoder.train()
            self._freeze_vae_encoder = False
            logger.info("VAE encoder unfrozen.")

    def freeze_vae_decoder(self) -> None:
        self.decoder_adapter._freeze_body()
        self._freeze_vae_decoder = True
        logger.info("VAE decoder frozen.")

    def unfreeze_vae_decoder(self) -> None:
        self.decoder_adapter._unfreeze_body()
        self._freeze_vae_decoder = False
        logger.info("VAE decoder unfrozen.")

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing on DiT transformer blocks.

        Trades ~20-30% slower forward for ~40% lower activation memory,
        allowing larger batch sizes or patch sizes.
        """
        if hasattr(self.dit, "enable_gradient_checkpointing"):
            self.dit.enable_gradient_checkpointing()
            self._gradient_checkpointing = True
            logger.info("Gradient checkpointing enabled (diffusers API).")
            return

        block_container = None
        for attr in ("transformer_blocks", "blocks", "layers"):
            if hasattr(self.dit, attr):
                block_container = getattr(self.dit, attr)
                break

        if block_container is None:
            logger.warning(
                "Cannot find transformer block container on %s — "
                "gradient checkpointing not applied.",
                type(self.dit).__name__,
            )
            return

        for block in block_container:
            original_forward = block.forward

            def _make_ckpt_forward(fwd):
                def ckpt_forward(*args, **kwargs):
                    if not torch.is_grad_enabled():
                        return fwd(*args, **kwargs)
                    return torch.utils.checkpoint.checkpoint(
                        fwd, *args, use_reentrant=False, **kwargs,
                    )
                return ckpt_forward

            block.forward = _make_ckpt_forward(original_forward)
            block._original_forward = original_forward

        self._gradient_checkpointing = True
        logger.info(
            "Gradient checkpointing enabled (manual, %d blocks).",
            len(block_container),
        )

    def disable_gradient_checkpointing(self) -> None:
        """Disable activation checkpointing, restoring original block forwards."""
        if hasattr(self.dit, "disable_gradient_checkpointing"):
            self.dit.disable_gradient_checkpointing()
            self._gradient_checkpointing = False
            logger.info("Gradient checkpointing disabled (diffusers API).")
            return

        block_container = None
        for attr in ("transformer_blocks", "blocks", "layers"):
            if hasattr(self.dit, attr):
                block_container = getattr(self.dit, attr)
                break

        if block_container is not None:
            for block in block_container:
                if hasattr(block, "_original_forward"):
                    block.forward = block._original_forward
                    del block._original_forward

        self._gradient_checkpointing = False
        logger.info("Gradient checkpointing disabled.")

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
# 3-D fit verification
# ---------------------------------------------------------------------------

def verify_fit(
    variant: str = "2B",
    input_shape: Tuple[int, ...] = (1, 1, 32, 64, 64),
    num_classes: int = 16,
    emb_dim: int = 16,
    feature_size: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Verify a Cosmos-Predict2.5 variant for 3-stage volumetric segmentation.

    Checks input/output dimensionality, sequence-length constraints,
    estimated memory footprint, and tokeniser compression requirements.

    Args:
        variant: ``"2B"`` or ``"14B"``.
        input_shape: ``(B, C, D, H, W)`` test shape.
        num_classes: Semantic classes.
        emb_dim: Instance embedding dim.
        feature_size: Intermediate feature channels.
        device: Target device for memory estimation.

    Returns:
        Dict with ``compatible``, ``warnings``, ``errors``, ``checks``.
    """
    variant = variant.upper()
    if variant not in _VARIANT_CONFIGS:
        return {
            "compatible": False,
            "errors": [f"Unknown variant: {variant}"],
            "warnings": [],
            "checks": {},
        }

    cfg = _VARIANT_CONFIGS[variant]
    B, _C, D, H, W = input_shape
    results: Dict[str, Any] = {
        "variant": variant,
        "compatible": True,
        "warnings": [],
        "errors": [],
        "checks": {},
    }

    s, t = cfg.spatial_compression, cfg.temporal_compression
    D_lat = D // t
    H_lat = H // s
    W_lat = W // s
    results["checks"]["latent_spatial"] = (D_lat, H_lat, W_lat)

    if D_lat < 1 or H_lat < 1 or W_lat < 1:
        results["errors"].append(
            f"Input {D}x{H}x{W} too small for "
            f"temporal_compression={t} / spatial_compression={s}. "
            f"Minimum input: {t}x{s}x{s}."
        )
        results["compatible"] = False

    if D % t != 0 or H % s != 0 or W % s != 0:
        results["warnings"].append(
            f"Input {D}x{H}x{W} not evenly divisible by compression. "
            f"Padding will be applied."
        )

    P = cfg.patch_size
    D_p = D_lat + (P - D_lat % P) % P
    H_p = H_lat + (P - H_lat % P) % P
    W_p = W_lat + (P - W_lat % P) % P
    seq_len = (D_p // P) * (H_p // P) * (W_p // P)
    results["checks"]["sequence_length"] = seq_len
    results["checks"]["max_sequence_length"] = cfg.max_sequence_length

    if seq_len > cfg.max_sequence_length:
        results["errors"].append(
            f"Sequence length {seq_len} exceeds max "
            f"{cfg.max_sequence_length}.  Reduce input volume size."
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
    head_params = (feature_size * 64 + 64 * (num_classes + emb_dim + geom_ch)) * 27
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
        available_gb = (
            torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        )
        results["checks"]["memory_estimate_gb"]["available"] = round(
            available_gb, 2,
        )
        if total_gb > available_gb * 0.85:
            results["warnings"].append(
                f"Estimated {total_gb:.1f} GB may exceed 85% of "
                f"available {available_gb:.1f} GB GPU memory."
            )
    else:
        results["warnings"].append(
            "CUDA not available -- cannot check GPU memory."
        )

    if variant == "14B":
        results["warnings"].append(
            "14B variant requires multi-GPU or CPU offloading for "
            "3-D volumes.  Use gradient checkpointing and mixed precision."
        )

    results["checks"]["stage_compatibility"] = {
        "stage1_semantic": (
            "Compatible -- standard voxel classification via logit head."
        ),
        "stage2_instance": (
            "Compatible -- embedding head produces per-voxel vectors."
        ),
        "stage3_geometry": (
            "Compatible with caveats -- Cosmos backbone is trained on "
            "natural video, not EM volumes.  Direction / covariance "
            "heads are randomly initialised.  The 3-D geometry head "
            "outputs S + S**2 + 4 = 16 channels (dir=3 + cov=9 + rgba=4)."
        ),
    }

    if not results["errors"]:
        logger.info(
            "verify_fit(%s, 3D): PASS -- %d warnings",
            variant, len(results["warnings"]),
        )
    else:
        results["compatible"] = False
        logger.warning("verify_fit(%s, 3D): FAIL -- %s", variant, results["errors"])

    return results
