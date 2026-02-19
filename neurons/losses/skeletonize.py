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
        self._expanded_dims = False

        self.endpoint_check = self._single_neighbor_check
        if simple_point_detection == "Boolean":
            self.simple_check = self._boolean_simple_check
        elif simple_point_detection == "EulerCharacteristic":
            self.simple_check = self._euler_characteristic_simple_check
        else:
            raise ValueError(
                f"Unknown simple_point_detection: {simple_point_detection!r}"
            )

    def forward(self, img):
        img = self._prepare_input(img)
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

        return self._prepare_output(img)

    # ------------------------------------------------------------------

    def _prepare_input(self, img):
        if img.dim() == 5:
            self._expanded_dims = False
        elif img.dim() == 4:
            self._expanded_dims = True
            img = img.unsqueeze(2)
        else:
            raise ValueError(
                "Expected 4-D [B,1,H,W] or 5-D [B,1,D,H,W] input, "
                f"got {img.dim()}-D."
            )
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Image values must lie in [0, 1].")
        return F.pad(img, (1, 1, 1, 1, 1, 1), value=0)

    def _prepare_output(self, img):
        img = img[:, :, 1:-1, 1:-1, 1:-1]
        if self._expanded_dims:
            img = img.squeeze(2)
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
        K = torch.tensor(
            [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
            device=img.device, dtype=img.dtype,
        ).view(1, 1, 3, 3, 3)
        n26 = F.conv3d(img, K)
        return F.hardtanh(-(n26 - 2), min_val=0, max_val=1)

    # ------------------------------------------------------------------
    # Simple-point detection: Boolean (Bertrand 1996)
    # ------------------------------------------------------------------

    def _boolean_simple_check(self, img):
        img = F.pad(img, (1, 1, 1, 1, 1, 1), value=0)
        x = 2.0 * img - 1.0

        K_N6 = torch.tensor(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
             [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
            device=img.device, dtype=img.dtype,
        ).view(1, 1, 3, 3, 3)
        n6z = F.conv3d(1 - img, K_N6, stride=2)
        c1 = (F.hardtanh(n6z, 0, 1) * F.hardtanh(-(n6z - 2), 0, 1))

        K_N26 = torch.tensor(
            [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
            device=img.device, dtype=img.dtype,
        ).view(1, 1, 3, 3, 3)
        n26 = F.conv3d(img, K_N26, stride=2)
        c2 = (F.hardtanh(n26, 0, 1) * F.hardtanh(-(n26 - 2), 0, 1))

        K_N18 = torch.tensor(
            [[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
             [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
             [[0, 1, 0], [1, 1, 1], [0, 1, 0]]],
            device=img.device, dtype=img.dtype,
        ).view(1, 1, 3, 3, 3)
        n18 = F.conv3d(img, K_N18, stride=2)
        c3a = (F.hardtanh(n18, 0, 1) * F.hardtanh(-(n18 - 2), 0, 1))

        K_B26 = torch.tensor(
            [[[1, -1, 0], [-1, -1, 0], [0, 0, 0]],
             [[-1, -1, 0], [-1, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            device=img.device, dtype=img.dtype,
        ).view(1, 1, 3, 3, 3)
        _flip_dims_list = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        b26 = sum(
            F.relu(F.conv3d(x, torch.flip(K_B26, dims=d) if d else K_B26, stride=2) - 6)
            for d in _flip_dims_list
        )
        c3 = c3a * F.hardtanh(-(b26 - 1), 0, 1)

        K_A6 = torch.tensor(
            [[[0, 1, 0], [1, -1, 1], [0, 1, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            device=img.device, dtype=img.dtype,
        ).view(1, 1, 3, 3, 3)
        a6_kernels = [
            K_A6,
            torch.rot90(K_A6, dims=[2, 3]),
            torch.rot90(K_A6, dims=[2, 4]),
            torch.flip(K_A6, dims=[2]),
            torch.rot90(torch.flip(K_A6, dims=[2]), dims=[2, 3]),
            torch.rot90(torch.flip(K_A6, dims=[2]), dims=[2, 4]),
        ]
        a6 = sum(F.relu(F.conv3d(x, k, stride=2) - 4) for k in a6_kernels)
        c4a = F.hardtanh(-(a6 - 1), 0, 1)
        c4b = F.hardtanh(-(b26 - 1), 0, 1)

        K_B18 = torch.tensor(
            [[[0, 1, 0], [-1, -1, -1], [0, 0, 0]],
             [[-1, -1, -1], [-1, 0, -1], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            device=img.device, dtype=img.dtype,
        ).view(1, 1, 3, 3, 3)
        b18_kernels = [K_B18]
        for k in range(1, 4):
            b18_kernels.append(torch.rot90(K_B18, dims=[2, 4], k=k))
        b18_kernels.append(torch.rot90(K_B18, dims=[3, 4]))
        for k in range(1, 4):
            b18_kernels.append(
                torch.rot90(torch.rot90(K_B18, dims=[3, 4]), dims=[2, 4], k=k)
            )
        b18_kernels.append(torch.rot90(K_B18, dims=[3, 4], k=2))
        for k in range(1, 4):
            b18_kernels.append(
                torch.rot90(
                    torch.rot90(K_B18, dims=[3, 4], k=2), dims=[2, 4], k=k
                )
            )
        b18 = sum(F.relu(F.conv3d(x, k, stride=2) - 8) for k in b18_kernels)
        c4c = F.hardtanh(-(b18 - 1), 0, 1)

        K_A18 = torch.tensor(
            [[[0, -1, 0], [0, -1, 0], [0, 0, 0]],
             [[0, -1, 0], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            device=img.device, dtype=img.dtype,
        ).view(1, 1, 3, 3, 3)
        a18_kernels = [K_A18]
        for k in range(1, 4):
            a18_kernels.append(torch.rot90(K_A18, dims=[2, 4], k=k))
        a18_kernels.append(torch.rot90(K_A18, dims=[3, 4]))
        for k in range(1, 4):
            a18_kernels.append(
                torch.rot90(torch.rot90(K_A18, dims=[3, 4]), dims=[2, 4], k=k)
            )
        a18_kernels.append(torch.rot90(K_A18, dims=[3, 4], k=2))
        for k in range(1, 4):
            a18_kernels.append(
                torch.rot90(
                    torch.rot90(K_A18, dims=[3, 4], k=2), dims=[2, 4], k=k
                )
            )
        a18 = sum(F.relu(F.conv3d(x, k, stride=2) - 2) for k in a18_kernels)

        K_A26 = torch.tensor(
            [[[-1, -1, 0], [-1, -1, 0], [0, 0, 0]],
             [[-1, -1, 0], [-1, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            device=img.device, dtype=img.dtype,
        ).view(1, 1, 3, 3, 3)
        a26 = sum(
            F.relu(
                F.conv3d(x, torch.flip(K_A26, dims=d) if d else K_A26, stride=2) - 6
            )
            for d in _flip_dims_list
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

        mask = torch.ones_like(img)
        mask[:, :, 1::2, 1::2, 1::2] = 0
        masked_img = img.clone() * mask
        inv = -(2.0 * img - 1.0)
        inv_m = -(2.0 * masked_img - 1.0)

        nv = F.avg_pool3d(F.relu(inv), 3, stride=2) * 27
        nv_m = F.avg_pool3d(F.relu(inv_m), 3, stride=2) * 27

        Ke = {
            "ud": torch.tensor([0.5, 0.5], device=img.device, dtype=img.dtype).view(1, 1, 2, 1, 1),
            "ns": torch.tensor([0.5, 0.5], device=img.device, dtype=img.dtype).view(1, 1, 1, 2, 1),
            "we": torch.tensor([0.5, 0.5], device=img.device, dtype=img.dtype).view(1, 1, 1, 1, 2),
        }
        pool_e = {"ud": (2, 3, 3), "ns": (3, 2, 3), "we": (3, 3, 2)}
        ne = sum(F.avg_pool3d(F.relu(F.conv3d(inv, k)), pool_e[n], stride=2) * 18 for n, k in Ke.items())
        ne_m = sum(F.avg_pool3d(F.relu(F.conv3d(inv_m, k)), pool_e[n], stride=2) * 18 for n, k in Ke.items())

        Kf = {
            "ud": torch.tensor([[.25, .25], [.25, .25]], device=img.device, dtype=img.dtype).view(1, 1, 1, 2, 2),
            "ns": torch.tensor([[.25, .25], [.25, .25]], device=img.device, dtype=img.dtype).view(1, 1, 2, 1, 2),
            "we": torch.tensor([[.25, .25], [.25, .25]], device=img.device, dtype=img.dtype).view(1, 1, 2, 2, 1),
        }
        pool_f = {"ud": (3, 2, 2), "ns": (2, 3, 2), "we": (2, 2, 3)}
        nf = sum(F.avg_pool3d(F.relu(F.conv3d(inv, k) - 0.5) * 2, pool_f[n], stride=2) * 12 for n, k in Kf.items())
        nf_m = sum(F.avg_pool3d(F.relu(F.conv3d(inv_m, k) - 0.5) * 2, pool_f[n], stride=2) * 12 for n, k in Kf.items())

        Ko = torch.full((1, 1, 2, 2, 2), 0.125, device=img.device, dtype=img.dtype)
        no_ = F.avg_pool3d(F.relu(F.conv3d(inv, Ko) - 0.75) * 4, 2, stride=2) * 8
        no_m = F.avg_pool3d(F.relu(F.conv3d(inv_m, Ko) - 0.75) * 4, 2, stride=2) * 8

        chi = nv - ne + nf - no_
        chi_m = nv_m - ne_m + nf_m - no_m

        change = F.hardtanh(torch.abs(chi_m - chi), min_val=0, max_val=1)
        is_simple = 1 - change
        return (is_simple.detach() > 0.5).float() - is_simple.detach() + is_simple
