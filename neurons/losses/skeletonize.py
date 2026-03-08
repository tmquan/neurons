"""Differentiable morphological skeletonization (PyTorch).

Vendored from:
    https://github.com/martinmenten/skeletonization-for-gradient-based-optimization

Reference:
    Martin J. Menten et al.  "A skeletonization algorithm for gradient-based
    optimization."  ICCV 2023.

Supports 2-D [B, 1, H, W] and 3-D [B, 1, D, H, W] binary inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _make_flip_variants(base_kernel):
    """Generate 8 flip variants for 3x3x3 kernels (all axis-flip combos)."""
    flip_dims_list = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
    return [torch.flip(base_kernel, dims=d) if d else base_kernel.clone() for d in flip_dims_list]


def _make_b18_rot_variants(base_kernel):
    """Generate 12 rotation variants for B18/A18 kernels."""
    kernels = [base_kernel.clone()]
    for k in range(1, 4):
        kernels.append(torch.rot90(base_kernel, dims=[2, 4], k=k))
    rot1 = torch.rot90(base_kernel, dims=[3, 4])
    kernels.append(rot1)
    for k in range(1, 4):
        kernels.append(torch.rot90(rot1, dims=[2, 4], k=k))
    rot2 = torch.rot90(base_kernel, dims=[3, 4], k=2)
    kernels.append(rot2)
    for k in range(1, 4):
        kernels.append(torch.rot90(rot2, dims=[2, 4], k=k))
    return kernels


class Skeletonize(nn.Module):
    """Iterative boundary-peeling skeletonization as a torch.nn.Module.

    Args:
        probabilistic: binarise soft inputs via reparameterization trick.
            Set False for already-binary masks.
        beta: logistic-noise scale for the reparameterization trick.
        tau: Boltzmann temperature for the reparameterization trick.
        simple_point_detection: "Boolean" (Bertrand 1996) or
            "EulerCharacteristic" (Lobregt 1980).
        num_iter: peeling iterations (each contains 8 subfield passes).
            Roughly one boundary-pixel layer is removed per iteration; set
            to at least the expected maximum inscribed-ball radius.
    """

    def __init__(
        self,
        probabilistic=True,
        beta=0.33,
        tau=1.0,
        simple_point_detection="Boolean",
        num_iter=5,
    ):
        super().__init__()
        self.probabilistic = probabilistic
        self.tau = tau
        self.beta = beta
        self.num_iter = num_iter

        self.endpoint_check = self._single_neighbor_check
        if simple_point_detection == "Boolean":
            self.simple_check = self._boolean_simple_check
        elif simple_point_detection == "EulerCharacteristic":
            self.simple_check = self._euler_characteristic_simple_check
        else:
            raise ValueError(
                f"Unknown simple_point_detection: {simple_point_detection!r}"
            )

        self._register_neighbor_kernel()
        self._register_boolean_kernels()
        self._register_euler_kernels()

    # ------------------------------------------------------------------
    # Kernel registration
    # ------------------------------------------------------------------

    def _register_neighbor_kernel(self):
        K = torch.tensor(
            [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)
        self.register_buffer("_K_neighbor", K)

    def _register_boolean_kernels(self):
        K_N6 = torch.tensor(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
             [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)
        self.register_buffer("_K_N6", K_N6)

        K_N26 = torch.tensor(
            [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)
        self.register_buffer("_K_N26", K_N26)

        K_N18 = torch.tensor(
            [[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
             [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
             [[0, 1, 0], [1, 1, 1], [0, 1, 0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)
        self.register_buffer("_K_N18", K_N18)

        K_B26 = torch.tensor(
            [[[1, -1, 0], [-1, -1, 0], [0, 0, 0]],
             [[-1, -1, 0], [-1, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)
        b26_variants = _make_flip_variants(K_B26)
        for i, k in enumerate(b26_variants):
            self.register_buffer(f"_K_B26_{i}", k)
        self._n_B26 = len(b26_variants)

        K_A6 = torch.tensor(
            [[[0, 1, 0], [1, -1, 1], [0, 1, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)
        a6_kernels = [
            K_A6,
            torch.rot90(K_A6, dims=[2, 3]),
            torch.rot90(K_A6, dims=[2, 4]),
            torch.flip(K_A6, dims=[2]),
            torch.rot90(torch.flip(K_A6, dims=[2]), dims=[2, 3]),
            torch.rot90(torch.flip(K_A6, dims=[2]), dims=[2, 4]),
        ]
        for i, k in enumerate(a6_kernels):
            self.register_buffer(f"_K_A6_{i}", k)
        self._n_A6 = len(a6_kernels)

        K_B18 = torch.tensor(
            [[[0, 1, 0], [-1, -1, -1], [0, 0, 0]],
             [[-1, -1, -1], [-1, 0, -1], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)
        b18_kernels = _make_b18_rot_variants(K_B18)
        for i, k in enumerate(b18_kernels):
            self.register_buffer(f"_K_B18_{i}", k)
        self._n_B18 = len(b18_kernels)

        K_A18 = torch.tensor(
            [[[0, -1, 0], [0, -1, 0], [0, 0, 0]],
             [[0, -1, 0], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)
        a18_kernels = _make_b18_rot_variants(K_A18)
        for i, k in enumerate(a18_kernels):
            self.register_buffer(f"_K_A18_{i}", k)
        self._n_A18 = len(a18_kernels)

        K_A26 = torch.tensor(
            [[[-1, -1, 0], [-1, -1, 0], [0, 0, 0]],
             [[-1, -1, 0], [-1, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3, 3)
        a26_variants = _make_flip_variants(K_A26)
        for i, k in enumerate(a26_variants):
            self.register_buffer(f"_K_A26_{i}", k)
        self._n_A26 = len(a26_variants)

    def _register_euler_kernels(self):
        self.register_buffer(
            "_Ke_ud",
            torch.tensor([0.5, 0.5], dtype=torch.float32).view(1, 1, 2, 1, 1),
        )
        self.register_buffer(
            "_Ke_ns",
            torch.tensor([0.5, 0.5], dtype=torch.float32).view(1, 1, 1, 2, 1),
        )
        self.register_buffer(
            "_Ke_we",
            torch.tensor([0.5, 0.5], dtype=torch.float32).view(1, 1, 1, 1, 2),
        )

        self.register_buffer(
            "_Kf_ud",
            torch.tensor([[.25, .25], [.25, .25]], dtype=torch.float32).view(1, 1, 1, 2, 2),
        )
        self.register_buffer(
            "_Kf_ns",
            torch.tensor([[.25, .25], [.25, .25]], dtype=torch.float32).view(1, 1, 2, 1, 2),
        )
        self.register_buffer(
            "_Kf_we",
            torch.tensor([[.25, .25], [.25, .25]], dtype=torch.float32).view(1, 1, 2, 2, 1),
        )

        self.register_buffer(
            "_Ko",
            torch.full((1, 1, 2, 2, 2), 0.125, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Helpers to retrieve kernel lists
    # ------------------------------------------------------------------

    def _get_buffers(self, prefix, count):
        return [getattr(self, f"{prefix}{i}") for i in range(count)]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img):
        img, expanded = self._prepare_input(img)
        if self.probabilistic:
            img = self._stochastic_discretization(img)

        for _ in range(self.num_iter):
            is_endpoint = self.endpoint_check(img)
            x_off = [0, 1, 0, 1, 0, 1, 0, 1]
            y_off = [0, 0, 1, 1, 0, 0, 1, 1]
            z_off = [0, 0, 0, 0, 1, 1, 1, 1]
            for xo, yo, zo in zip(x_off, y_off, z_off):
                is_simple = self.simple_check(img[:, :, xo:, yo:, zo:])
                deletion = is_simple * (
                    1 - is_endpoint[:, :, xo::2, yo::2, zo::2]
                )
                img[:, :, xo::2, yo::2, zo::2] = torch.min(
                    img[:, :, xo::2, yo::2, zo::2].clone(),
                    1 - deletion,
                )

        return self._prepare_output(img, expanded)

    # ------------------------------------------------------------------

    def _prepare_input(self, img):
        expanded = False
        if img.dim() == 5:
            pass
        elif img.dim() == 4:
            expanded = True
            img = rearrange(img, "b c h w -> b c 1 h w")
        else:
            raise ValueError(
                "Expected 4-D [B,1,H,W] or 5-D [B,1,D,H,W] input, "
                f"got {img.dim()}-D."
            )
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Image values must lie in [0, 1].")
        return F.pad(img, (1, 1, 1, 1, 1, 1), value=0), expanded

    def _prepare_output(self, img, expanded=False):
        img = img[:, :, 1:-1, 1:-1, 1:-1]
        if expanded:
            img = rearrange(img, "b c 1 h w -> b c h w")
        return img

    def _stochastic_discretization(self, img):
        alpha = (img + 1e-8) / (1.0 - img + 1e-8)
        uniform_noise = torch.empty_like(img).uniform_(1e-8, 1 - 1e-8)
        logistic_noise = torch.log(uniform_noise) - torch.log(1 - uniform_noise)
        img = torch.sigmoid(
            (torch.log(alpha) + logistic_noise * self.beta) / self.tau
        )
        return (img.detach() > 0.5).float() - img.detach() + img

    # ------------------------------------------------------------------
    # Endpoint detection
    # ------------------------------------------------------------------

    def _single_neighbor_check(self, img):
        img = F.pad(img, (1, 1, 1, 1, 1, 1))
        K = self._K_neighbor.to(dtype=img.dtype)
        n26 = F.conv3d(img, K)
        return F.hardtanh(-(n26 - 2), min_val=0, max_val=1)

    # ------------------------------------------------------------------
    # Simple-point detection: Boolean (Bertrand 1996)
    # ------------------------------------------------------------------

    def _boolean_simple_check(self, img):
        img = F.pad(img, (1, 1, 1, 1, 1, 1), value=0)
        x = 2.0 * img - 1.0
        dt = img.dtype

        n6z = F.conv3d(1 - img, self._K_N6.to(dtype=dt), stride=2)
        c1 = (F.hardtanh(n6z, 0, 1) * F.hardtanh(-(n6z - 2), 0, 1))

        n26 = F.conv3d(img, self._K_N26.to(dtype=dt), stride=2)
        c2 = (F.hardtanh(n26, 0, 1) * F.hardtanh(-(n26 - 2), 0, 1))

        n18 = F.conv3d(img, self._K_N18.to(dtype=dt), stride=2)
        c3a = (F.hardtanh(n18, 0, 1) * F.hardtanh(-(n18 - 2), 0, 1))

        b26_kernels = self._get_buffers("_K_B26_", self._n_B26)
        b26 = sum(
            F.relu(F.conv3d(x, k.to(dtype=dt), stride=2) - 6)
            for k in b26_kernels
        )
        c3 = c3a * F.hardtanh(-(b26 - 1), 0, 1)

        a6_kernels = self._get_buffers("_K_A6_", self._n_A6)
        a6 = sum(F.relu(F.conv3d(x, k.to(dtype=dt), stride=2) - 4) for k in a6_kernels)
        c4a = F.hardtanh(-(a6 - 1), 0, 1)
        c4b = F.hardtanh(-(b26 - 1), 0, 1)

        b18_kernels = self._get_buffers("_K_B18_", self._n_B18)
        b18 = sum(F.relu(F.conv3d(x, k.to(dtype=dt), stride=2) - 8) for k in b18_kernels)
        c4c = F.hardtanh(-(b18 - 1), 0, 1)

        a18_kernels = self._get_buffers("_K_A18_", self._n_A18)
        a18 = sum(F.relu(F.conv3d(x, k.to(dtype=dt), stride=2) - 2) for k in a18_kernels)

        a26_kernels = self._get_buffers("_K_A26_", self._n_A26)
        a26 = sum(
            F.relu(F.conv3d(x, k.to(dtype=dt), stride=2) - 6)
            for k in a26_kernels
        )

        chi = n6z - a18 + a26
        c4d = (F.hardtanh(chi, 0, 1) * F.hardtanh(-(chi - 2), 0, 1))
        c4 = c4a * c4b * c4c * c4d

        combined = torch.cat([c1, c2, c3, c4], dim=1)
        return torch.amax(combined, dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # Simple-point detection: Euler characteristic (Lobregt 1980)
    # ------------------------------------------------------------------

    def _euler_characteristic_simple_check(self, img):
        img = F.pad(img, (1, 1, 1, 1, 1, 1), value=0)
        dt = img.dtype

        mask = torch.ones_like(img)
        mask[:, :, 1::2, 1::2, 1::2] = 0
        masked_img = img.clone() * mask
        inv = -(2.0 * img - 1.0)
        inv_m = -(2.0 * masked_img - 1.0)

        nv = F.avg_pool3d(F.relu(inv), 3, stride=2) * 27
        nv_m = F.avg_pool3d(F.relu(inv_m), 3, stride=2) * 27

        Ke_items = [
            ("_Ke_ud", (2, 3, 3)),
            ("_Ke_ns", (3, 2, 3)),
            ("_Ke_we", (3, 3, 2)),
        ]
        ne = sum(
            F.avg_pool3d(F.relu(F.conv3d(inv, getattr(self, name).to(dtype=dt))), pool, stride=2) * 18
            for name, pool in Ke_items
        )
        ne_m = sum(
            F.avg_pool3d(F.relu(F.conv3d(inv_m, getattr(self, name).to(dtype=dt))), pool, stride=2) * 18
            for name, pool in Ke_items
        )

        Kf_items = [
            ("_Kf_ud", (3, 2, 2)),
            ("_Kf_ns", (2, 3, 2)),
            ("_Kf_we", (2, 2, 3)),
        ]
        nf = sum(
            F.avg_pool3d(F.relu(F.conv3d(inv, getattr(self, name).to(dtype=dt)) - 0.5) * 2, pool, stride=2) * 12
            for name, pool in Kf_items
        )
        nf_m = sum(
            F.avg_pool3d(F.relu(F.conv3d(inv_m, getattr(self, name).to(dtype=dt)) - 0.5) * 2, pool, stride=2) * 12
            for name, pool in Kf_items
        )

        Ko = self._Ko.to(dtype=dt)
        no_ = F.avg_pool3d(F.relu(F.conv3d(inv, Ko) - 0.75) * 4, 2, stride=2) * 8
        no_m = F.avg_pool3d(F.relu(F.conv3d(inv_m, Ko) - 0.75) * 4, 2, stride=2) * 8

        chi = nv - ne + nf - no_
        chi_m = nv_m - ne_m + nf_m - no_m

        change = F.hardtanh(torch.abs(chi_m - chi), min_val=0, max_val=1)
        is_simple = 1 - change
        return (is_simple.detach() > 0.5).float() - is_simple.detach() + is_simple
