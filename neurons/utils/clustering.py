"""
Embedding / offset clustering utilities for connectomics segmentation.

Public API
----------
- ``cluster_embeddings``      -- unified entry point; switch algorithm via
                                 ``algorithm={"meanshift", "hdbscan",
                                 "soft_meanshift", "spatial_cc"}``.  Picks
                                 the fastest available backend (GPU → CPU).
- ``cluster_spatial_cc``      -- connected components on the spatial-neighbour
                                 embedding-affinity graph.  GPU via cupyx.
- ``cluster_offsets_hough``   -- Hough voting on predicted spatial offsets.

Back-compatible thin wrappers (kept so existing call sites keep working):
- ``cluster_embeddings_meanshift``  -> ``cluster_embeddings(algorithm="meanshift")``
- ``cluster_embeddings_soft``       -> ``cluster_embeddings(algorithm="soft_meanshift")``
- ``cluster_embeddings_hdbscan``    -> ``cluster_embeddings(algorithm="hdbscan")``
- ``cluster_embeddings_spatial_cc`` -> ``cluster_embeddings(algorithm="spatial_cc")``

Backend selection (per algorithm, in preference order):

- ``meanshift``       : cuML ``MeanShift`` (GPU) → sklearn ``MeanShift`` (CPU).
- ``hdbscan``         : cuML ``HDBSCAN`` (GPU) → ``hdbscan.HDBSCAN`` (CPU C impl)
                         → sklearn ``HDBSCAN`` (CPU, requires sklearn >= 1.3).
- ``soft_meanshift``  : differentiable torch implementation
                         (:class:`neurons.inference.clusterer.SoftMeanShift`);
                         runs on whatever device the input tensor lives on.
- ``spatial_cc``      : ``cupyx.scipy.sparse.csgraph.connected_components``
                         (GPU, zero-copy via DLPack) → ``scipy.sparse.csgraph``
                         (CPU).  Neither cuml nor cucim provides sparse CC.

All algorithms return an integer label tensor with the same spatial shape as
the input, where ``0`` is background / noise and foreground instances are
numbered ``1..K``.  Only ``soft_meanshift`` is differentiable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, reduce


# ---------------------------------------------------------------------------
# Dimension-agnostic einops helpers
# ---------------------------------------------------------------------------


def _spatial_axes(spatial_shape: Tuple[int, ...]) -> Tuple[str, Dict[str, int]]:
    """Build an einops pattern snippet ("sA sB ...") and axis-size dict.

    Used to round-trip between flat and arbitrary-rank spatial layouts
    without losing dimension semantics (e.g. ``"... (sA sB sC) -> ... sA sB sC"``).
    Axis names use ASCII letters because einops rejects leading/trailing
    underscores and digit-only identifiers.
    """
    import string
    names_list = [f"s{string.ascii_uppercase[i]}" for i in range(len(spatial_shape))]
    axes = {name: int(s) for name, s in zip(names_list, spatial_shape)}
    names = " ".join(names_list)
    return names, axes


def _reshape_to_spatial(flat: torch.Tensor, spatial_shape: Tuple[int, ...]) -> torch.Tensor:
    """Reshape the last axis of ``flat`` into ``*spatial_shape`` via einops."""
    names, axes = _spatial_axes(spatial_shape)
    if flat.dim() == 1:
        return rearrange(flat, f"({names}) -> {names}", **axes)
    return rearrange(flat, f"... ({names}) -> ... {names}", **axes)


# ---------------------------------------------------------------------------
# Backend probing (memoized)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _probe_cupy() -> Optional[Any]:
    """Return the ``cupy`` module if importable, else None."""
    try:
        import cupy as cp
    except Exception:
        return None
    return cp


@lru_cache(maxsize=1)
def _probe_cuml_hdbscan() -> Optional[Any]:
    """Return ``cuml.cluster.HDBSCAN`` if importable, else None."""
    if _probe_cupy() is None:
        return None
    try:
        from cuml.cluster import HDBSCAN as _CuHDBSCAN
    except Exception:
        return None
    return _CuHDBSCAN


@lru_cache(maxsize=1)
def _probe_cuml_meanshift() -> Optional[Any]:
    """Return ``cuml.cluster.MeanShift`` if importable, else None.

    Note: RAPIDS dropped MeanShift in cuML 23.x; on modern installs this
    probe will return None and MeanShift will fall back to sklearn.
    """
    if _probe_cupy() is None:
        return None
    try:
        from cuml.cluster import MeanShift as _CuMeanShift
    except Exception:
        return None
    return _CuMeanShift


@lru_cache(maxsize=1)
def _probe_hdbscan_pkg() -> Optional[Any]:
    """Return the standalone ``hdbscan`` package if installed, else None."""
    try:
        import hdbscan
    except Exception:
        return None
    return hdbscan


@lru_cache(maxsize=1)
def _probe_sklearn_hdbscan() -> Optional[Any]:
    """Return sklearn's ``HDBSCAN`` (>= 1.3) if available, else None."""
    try:
        from sklearn.cluster import HDBSCAN as _SKHDBSCAN
    except Exception:
        return None
    return _SKHDBSCAN


@lru_cache(maxsize=1)
def _probe_sklearn_meanshift() -> Optional[Any]:
    try:
        from sklearn.cluster import MeanShift as _SKMeanShift
    except Exception:
        return None
    return _SKMeanShift


@lru_cache(maxsize=1)
def _probe_scipy_csgraph() -> Optional[Any]:
    """Return :mod:`scipy.sparse.csgraph` if importable, else None.

    Required for the ``spatial_cc`` clusterer (CPU path).
    """
    try:
        from scipy.sparse import csgraph
    except Exception:
        return None
    return csgraph


@lru_cache(maxsize=1)
def _probe_scipy_sparse() -> Optional[Any]:
    try:
        from scipy import sparse
    except Exception:
        return None
    return sparse


@lru_cache(maxsize=1)
def _probe_cupy_csgraph() -> Optional[Tuple[Any, Any]]:
    """Return ``(cupyx.scipy.sparse, cupyx.scipy.sparse.csgraph)`` on GPU.

    Preferred fast path for ``spatial_cc``: ``connected_components`` runs
    fully on the device with no host round-trip.  Returns ``None`` if
    cupy is missing, the csgraph submodule is absent (cupy < 9), or the
    underlying ``pylibcugraph`` dependency is not installed (cupyx
    imports the submodule lazily but the CC call raises
    ``RuntimeError: pylibcugraph is not available`` without it — hence
    the probe executes a trivial 2-node call rather than trusting the
    import alone).
    """
    cp = _probe_cupy()
    if cp is None:
        return None
    try:
        from cupyx.scipy import sparse as cp_sparse
        from cupyx.scipy.sparse import csgraph as cp_csgraph
    except Exception:
        return None
    if not hasattr(cp_csgraph, "connected_components"):
        return None
    try:
        probe = cp_sparse.coo_matrix(
            (cp.ones(1, dtype=cp.float32),
             (cp.zeros(1, dtype=cp.int32), cp.ones(1, dtype=cp.int32))),
            shape=(2, 2),
        )
        cp_csgraph.connected_components(probe, directed=False)
    except Exception:
        return None
    return cp_sparse, cp_csgraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_ALGOS = ("meanshift", "hdbscan", "soft_meanshift", "spatial_cc")
_VALID_BACKENDS = ("auto", "cuml", "cupy", "hdbscan", "sklearn", "torch", "scipy")


def _as_fg_np(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor],
    normalize_embeddings: bool,
) -> Tuple[np.ndarray, torch.Tensor, Tuple[int, ...]]:
    """Flatten an ``[E, *spatial]`` embedding to foreground-only numpy.

    Returns
    -------
    emb_fg_np    : ``[N_fg, E]`` float32 numpy array of foreground embeddings.
    fg_idx       : ``[N_fg]`` long tensor of foreground indices into the
                    flattened spatial grid (on the original device).
    spatial_shape: original spatial shape.
    """
    if normalize_embeddings:
        import torch.nn.functional as F
        embedding = F.normalize(embedding, dim=0, eps=1e-6)

    spatial_shape = tuple(embedding.shape[1:])
    emb_flat = rearrange(embedding, "e ... -> (...) e")

    if foreground_mask is not None:
        fg_flat = rearrange(foreground_mask, "... -> (...)") > 0
    else:
        fg_flat = torch.ones(
            emb_flat.shape[0], dtype=torch.bool, device=embedding.device,
        )

    fg_idx = torch.where(fg_flat)[0]
    if len(fg_idx) == 0:
        return (
            np.zeros((0, emb_flat.shape[1]), dtype=np.float32),
            fg_idx,
            spatial_shape,
        )
    emb_fg_np = emb_flat[fg_idx].detach().cpu().to(torch.float32).numpy()
    return emb_fg_np, fg_idx, spatial_shape


def _as_fg_cupy(emb_fg_np: np.ndarray):
    """Move a ``[N_fg, E]`` numpy array onto the GPU as a cupy array."""
    cp = _probe_cupy()
    assert cp is not None, "cupy not available"
    return cp.asarray(emb_fg_np)


def _subsample(
    emb_fg: np.ndarray, max_points: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Uniformly subsample up to ``max_points`` rows from ``emb_fg``.

    Returns the subset and the integer indices used (or None if no
    subsampling was performed).
    """
    n = len(emb_fg)
    if n <= max_points:
        return emb_fg, None
    idx = rng.choice(n, size=max_points, replace=False)
    return emb_fg[idx], idx


def _propagate_labels(
    emb_fg: np.ndarray,
    sub_idx: np.ndarray,
    sub_labels: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Assign every foreground point to the nearest *surviving* cluster center.

    Points whose nearest-center distance exceeds ``epsilon`` are marked as
    noise (label 0 in the final 1-indexed space).
    """
    valid = sub_labels >= 0
    if not valid.any():
        return np.zeros(len(emb_fg), dtype=np.int64)

    uniq = np.unique(sub_labels[valid])
    K = len(uniq)
    centers = np.empty((K, emb_fg.shape[1]), dtype=np.float32)
    for k, u in enumerate(uniq):
        centers[k] = emb_fg[sub_idx][sub_labels == u].mean(axis=0)

    # Nearest-center assignment for all foreground points.
    # Chunked to keep memory under control on large volumes.
    labels = np.zeros(len(emb_fg), dtype=np.int64)
    chunk = 65_536
    for start in range(0, len(emb_fg), chunk):
        stop = min(start + chunk, len(emb_fg))
        d = np.linalg.norm(
            emb_fg[start:stop, None, :] - centers[None, :, :], axis=2,
        )
        nearest = d.argmin(axis=1)
        nearest_d = d[np.arange(stop - start), nearest]
        assign = nearest + 1  # 1-indexed; 0 reserved for background/noise
        if np.isfinite(epsilon) and epsilon > 0:
            assign = np.where(nearest_d <= epsilon, assign, 0)
        labels[start:stop] = assign
    return labels


def _remap_consecutive(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Remap foreground labels to ``1..K'`` after filtering tiny clusters."""
    uniq, counts = np.unique(labels, return_counts=True)
    fg_labels = uniq[uniq > 0]
    for u, c in zip(uniq, counts):
        if u > 0 and c < min_cluster_size:
            labels[labels == u] = 0
    uniq_fg = np.unique(labels[labels > 0])
    if len(uniq_fg) == 0:
        return labels.astype(np.int64)
    remap = np.zeros(int(uniq_fg.max()) + 1, dtype=np.int64)
    for new, old in enumerate(uniq_fg, start=1):
        remap[int(old)] = new
    out = labels.copy()
    out[out > 0] = remap[labels[labels > 0]]
    return out


# ---------------------------------------------------------------------------
# Per-algorithm dispatchers
# ---------------------------------------------------------------------------

def _run_meanshift(
    emb_fg: np.ndarray,
    bandwidth: float,
    bin_seeding: bool,
    backend: str,
) -> np.ndarray:
    """Return ``[N]`` int labels with ``-1`` marking noise."""
    cp = _probe_cupy()
    cuml_cls = _probe_cuml_meanshift() if backend in ("auto", "cuml") else None
    if cuml_cls is not None and cp is not None:
        try:
            emb_gpu = _as_fg_cupy(emb_fg)
            model = cuml_cls(bandwidth=bandwidth, bin_seeding=bin_seeding)
            labels_gpu = model.fit_predict(emb_gpu)
            return cp.asnumpy(labels_gpu).astype(np.int64)
        except Exception:
            if backend == "cuml":
                raise

    if backend == "cuml":
        raise RuntimeError(
            "backend='cuml' requested for MeanShift but cuML.cluster.MeanShift "
            "is not available (RAPIDS dropped MeanShift in cuML 23.x). "
            "Use backend='auto' or 'sklearn' for CPU MeanShift, or switch to "
            "algorithm='hdbscan' for a GPU-accelerated alternative."
        )

    sk_cls = _probe_sklearn_meanshift()
    if sk_cls is None:
        raise ImportError(
            "MeanShift requires scikit-learn (cuML no longer ships MeanShift)."
        )
    try:
        model = sk_cls(bandwidth=bandwidth, bin_seeding=bin_seeding)
        return np.asarray(model.fit_predict(emb_fg), dtype=np.int64)
    except ValueError:
        # Happens when every point collapses to one cluster on degenerate input.
        return np.zeros(len(emb_fg), dtype=np.int64)


def _run_hdbscan(
    emb_fg: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int],
    cluster_selection_epsilon: float,
    backend: str,
) -> np.ndarray:
    """Return ``[N]`` int labels with ``-1`` marking noise."""
    kw: Dict[str, Any] = dict(
        min_cluster_size=int(min_cluster_size),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
    )
    if min_samples is not None:
        kw["min_samples"] = int(min_samples)

    cp = _probe_cupy()
    cuml_cls = _probe_cuml_hdbscan() if backend in ("auto", "cuml") else None
    if cuml_cls is not None and cp is not None:
        try:
            emb_gpu = _as_fg_cupy(emb_fg)
            model = cuml_cls(**kw)
            labels_gpu = model.fit_predict(emb_gpu)
            return cp.asnumpy(labels_gpu).astype(np.int64)
        except Exception:
            if backend == "cuml":
                raise

    if backend == "cuml":
        raise RuntimeError(
            "backend='cuml' requested for HDBSCAN but cuML is not available."
        )

    if backend in ("auto", "hdbscan"):
        pkg = _probe_hdbscan_pkg()
        if pkg is not None:
            model = pkg.HDBSCAN(**kw)
            return np.asarray(model.fit_predict(emb_fg), dtype=np.int64)
        if backend == "hdbscan":
            raise ImportError(
                "backend='hdbscan' requested but the 'hdbscan' package is not installed."
            )

    sk_cls = _probe_sklearn_hdbscan()
    if sk_cls is None:
        raise ImportError(
            "HDBSCAN requires one of: cuML (GPU), the 'hdbscan' package, "
            "or scikit-learn >= 1.3."
        )
    model = sk_cls(**kw)
    return np.asarray(model.fit_predict(emb_fg), dtype=np.int64)


# ---------------------------------------------------------------------------
# Spatial connected-components on embedding-neighbour affinities
# ---------------------------------------------------------------------------

def _spatial_cc_edges(
    embedding: torch.Tensor,
    fg: torch.Tensor,
    delta_v: float,
    connectivity: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a sparse affinity edge list between spatial neighbours.

    Two foreground voxels ``i`` and ``j`` are linked iff they are spatial
    neighbours under the requested connectivity and their embedding
    distance is strictly less than ``delta_v``.

    Args:
        embedding: ``[E, *spatial]`` tensor (already L2-normalised if
            requested by the caller).
        fg: ``[*spatial]`` boolean foreground mask.
        delta_v: Euclidean threshold on per-edge embedding distance.
        connectivity: ``1`` for face-connectivity (6 in 3D, 4 in 2D);
            any other value enables full connectivity (26 in 3D, 8 in
            2D).  Matches scipy's ``connectivity`` convention.

    Returns:
        ``(src_flat, dst_flat)`` long tensors on the input's device,
        each of shape ``[M]``.  Indices are into the row-major flattened
        spatial grid (same convention as ``torch.Tensor.flatten()``).
    """
    spatial_shape = tuple(embedding.shape[1:])
    n_dims = len(spatial_shape)
    device = embedding.device

    # Row-major strides so (i_0, ..., i_{n-1}) -> sum_d i_d * stride_d
    strides = [1] * n_dims
    for i in range(n_dims - 2, -1, -1):
        strides[i] = strides[i + 1] * spatial_shape[i + 1]

    thr_sq = float(delta_v) ** 2

    # Enumerate unit offsets; for face connectivity keep only +ê_d, for
    # full connectivity keep every non-zero offset with first-non-zero
    # component positive (canonical half-space, avoids double counting).
    offsets: list[Tuple[int, ...]] = []
    if connectivity == 1:
        for d in range(n_dims):
            off = [0] * n_dims
            off[d] = 1
            offsets.append(tuple(off))
    else:
        import itertools
        for off in itertools.product((-1, 0, 1), repeat=n_dims):
            if all(o == 0 for o in off):
                continue
            first_nonzero = next(o for o in off if o != 0)
            if first_nonzero < 0:
                continue
            offsets.append(off)

    src_chunks: list[torch.Tensor] = []
    dst_chunks: list[torch.Tensor] = []

    for off in offsets:
        lo_slicer = [slice(None)] * n_dims
        hi_slicer = [slice(None)] * n_dims
        for d, o in enumerate(off):
            if o > 0:
                lo_slicer[d] = slice(0, spatial_shape[d] - o)
                hi_slicer[d] = slice(o, spatial_shape[d])
            elif o < 0:
                lo_slicer[d] = slice(-o, spatial_shape[d])
                hi_slicer[d] = slice(0, spatial_shape[d] + o)
        emb_lo = embedding[(slice(None), *lo_slicer)]
        emb_hi = embedding[(slice(None), *hi_slicer)]
        dist_sq = reduce((emb_hi - emb_lo) ** 2, "e ... -> ...", "sum")

        fg_lo = fg[tuple(lo_slicer)]
        fg_hi = fg[tuple(hi_slicer)]

        link = fg_lo & fg_hi & (dist_sq < thr_sq)
        if not bool(link.any()):
            continue

        pos = torch.nonzero(link, as_tuple=False)  # [M, n_dims] lo-positions
        src = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
        dst = torch.zeros_like(src)
        for d in range(n_dims):
            pos_d = pos[:, d]
            src = src + pos_d * strides[d]
            dst = dst + (pos_d + off[d]) * strides[d]
        src_chunks.append(src)
        dst_chunks.append(dst)

    if not src_chunks:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return torch.cat(src_chunks), torch.cat(dst_chunks)


def cluster_spatial_cc(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    *,
    delta_v: float = 0.5,
    min_cluster_size: int = 50,
    connectivity: int = 1,
    normalize_embeddings: bool = False,
    backend: str = "auto",
) -> torch.Tensor:
    """Cluster by connected components of the spatial-neighbour affinity graph.

    Two foreground voxels are linked iff they are spatial neighbours
    (face-connectivity by default) and their embedding vectors are
    within ``delta_v`` in Euclidean distance.  Connected components of
    the resulting sparse graph define instance labels.

    Compared to HDBSCAN / MeanShift / SoftMeanShift this variant:

    - **Respects spatial connectivity** — two unrelated cells with
      similar mean embeddings cannot merge because there is no chain
      of spatial neighbours between them.
    - **Scales linearly** in foreground voxels (``O(N · n_dims)``
      edges, ``O(N α(N))`` union-find).  No subsampling, no ``K``
      selection.
    - **Uses one threshold** — ``delta_v`` (= training's pull margin).
    - **Known failure mode**: a single mis-predicted embedding voxel
      at a cell-cell boundary can leak two cells into one component.
      Mitigate by tightening ``delta_v`` below the training margin,
      or by stacking a Mutex-Watershed post-process (not yet wired).

    Args:
        embedding: ``[E, *spatial]`` embedding tensor (unbatched).
        foreground_mask: Optional ``[*spatial]`` bool mask.  Background
            voxels always receive label 0.
        delta_v: Edge threshold.  Same semantics as the discriminative
            loss's pull margin; using the trained value is the right
            default.
        min_cluster_size: Clusters with fewer than this many voxels
            are dropped (mapped to label 0).  Applied at full resolution
            — no subsample bias.
        connectivity: ``1`` for face-connectivity (6 in 3D, 4 in 2D);
            any other value uses full connectivity (26 / 8).
        normalize_embeddings: L2-normalise before computing distances.
            Must match the flag used at training time.
        backend: ``"auto"`` (cupyx.scipy.sparse.csgraph on CUDA, else
            scipy) / ``"cupy"`` (force GPU) / ``"scipy"`` (force CPU).
            There is no ``cuml`` or ``cugraph`` path because neither
            library ships a sparse ``connected_components``; ``cupyx``
            covers this as of cupy >= 9.

    Returns:
        ``[*spatial]`` ``torch.long`` label tensor (0 = background,
        1..K = instances) on the same device as ``embedding``.
    """
    if backend not in ("auto", "cupy", "scipy"):
        raise ValueError(
            f"spatial_cc: unsupported backend {backend!r}; "
            f"choose 'auto', 'cupy', or 'scipy'."
        )

    device = embedding.device
    if embedding.dim() < 2:
        raise ValueError(
            f"cluster_spatial_cc expects an [E, *spatial] embedding; "
            f"got shape {tuple(embedding.shape)}."
        )

    if normalize_embeddings:
        import torch.nn.functional as F_
        embedding = F_.normalize(embedding, dim=0, eps=1e-6)

    spatial_shape = tuple(embedding.shape[1:])
    n_total = int(np.prod(spatial_shape)) if spatial_shape else 0

    if foreground_mask is None:
        fg = torch.ones(spatial_shape, dtype=torch.bool, device=device)
    else:
        fg = foreground_mask > 0
        if tuple(fg.shape) != spatial_shape:
            raise ValueError(
                f"foreground_mask shape {tuple(fg.shape)} does not match "
                f"embedding spatial shape {spatial_shape}."
            )

    labels_full = torch.zeros(n_total, device=device, dtype=torch.long)
    if n_total == 0 or not bool(fg.any()):
        return _reshape_to_spatial(labels_full, spatial_shape)

    # Build edges on the input's device (fast tensor ops in torch).
    src, dst = _spatial_cc_edges(embedding, fg, delta_v, connectivity)

    use_gpu = False
    gpu_mods = None
    if backend in ("auto", "cupy") and embedding.is_cuda:
        gpu_mods = _probe_cupy_csgraph()
        use_gpu = gpu_mods is not None
    if backend == "cupy" and not use_gpu:
        raise RuntimeError(
            "spatial_cc: backend='cupy' requires a CUDA embedding and "
            "cupyx.scipy.sparse.csgraph.connected_components (cupy >= 9)."
        )

    if use_gpu:
        labels_flat = _spatial_cc_run_gpu(
            src, dst, fg, n_total, gpu_mods, device,
        )
    else:
        csgraph = _probe_scipy_csgraph()
        sparse = _probe_scipy_sparse()
        if csgraph is None or sparse is None:
            raise ImportError(
                "spatial_cc CPU path requires scipy.sparse.csgraph."
            )
        labels_flat = _spatial_cc_run_cpu(
            src, dst, fg, n_total, sparse, csgraph, device,
        )

    # Filter small clusters and renumber to 1..K'.  Done on CPU because
    # `_remap_consecutive` is numpy-based; the tensor is already flat.
    raw = labels_flat.detach().cpu().numpy().astype(np.int64, copy=False)
    remapped = _remap_consecutive(raw, min_cluster_size)
    labels_full = torch.from_numpy(remapped).to(device=device, dtype=torch.long)
    return _reshape_to_spatial(labels_full, spatial_shape)


def _spatial_cc_run_cpu(
    src: torch.Tensor,
    dst: torch.Tensor,
    fg: torch.Tensor,
    n_total: int,
    sparse_mod: Any,
    csgraph_mod: Any,
    device: torch.device,
) -> torch.Tensor:
    """Run connected_components on CPU via scipy.sparse.csgraph."""
    src_np = src.detach().cpu().numpy().astype(np.int64, copy=False)
    dst_np = dst.detach().cpu().numpy().astype(np.int64, copy=False)
    data = np.ones(len(src_np), dtype=np.int8)
    graph = sparse_mod.coo_matrix(
        (data, (src_np, dst_np)), shape=(n_total, n_total),
    )
    _, comp = csgraph_mod.connected_components(graph, directed=False)
    comp_t = torch.from_numpy(comp.astype(np.int64, copy=False)).to(device)
    fg_flat = rearrange(fg, "... -> (...)")
    return (comp_t + 1) * fg_flat.to(comp_t.dtype)


def _spatial_cc_run_gpu(
    src: torch.Tensor,
    dst: torch.Tensor,
    fg: torch.Tensor,
    n_total: int,
    gpu_mods: Tuple[Any, Any],
    device: torch.device,
) -> torch.Tensor:
    """Run connected_components on GPU via cupyx.scipy.sparse.csgraph.

    Zero-copy handoff torch -> cupy via DLPack so the COO build and the
    CC pass both stay on the device.  Falls through to the scipy path
    via ImportError / RuntimeError at the call site on failure.
    """
    cp = _probe_cupy()
    assert cp is not None, "cupy must be available on the GPU path"
    cp_sparse, cp_csgraph = gpu_mods

    # torch -> cupy (zero-copy).  Data must be float32: cupyx's
    # `coo_matrix` constructor accepts {bool, float32, float64,
    # complex64, complex128} but the CC path internally invokes
    # cusparse's csc2csr which rejects bool / int; float32 is the
    # cheapest supported type.  Indices are int32 for the same reason
    # (cusparse requires 32-bit indices on current RAPIDS builds).
    src_cp = cp.from_dlpack(src.contiguous()).astype(cp.int32)
    dst_cp = cp.from_dlpack(dst.contiguous()).astype(cp.int32)
    data_cp = cp.ones(src_cp.size, dtype=cp.float32)

    graph = cp_sparse.coo_matrix(
        (data_cp, (src_cp, dst_cp)), shape=(n_total, n_total),
    )
    _, comp_cp = cp_csgraph.connected_components(graph, directed=False)

    # cupy -> torch (zero-copy) and combine with foreground mask.
    comp_t = torch.from_dlpack(comp_cp.astype(cp.int64)).to(device)
    fg_flat = rearrange(fg, "... -> (...)")
    return (comp_t + 1) * fg_flat.to(comp_t.dtype)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    algorithm: str = "meanshift",
    *,
    bandwidth: float = 0.5,
    min_cluster_size: int = 50,
    normalize_embeddings: bool = False,
    backend: str = "auto",
    # hdbscan
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: Optional[float] = None,
    max_points: int = 200_000,
    # meanshift
    bin_seeding: bool = True,
    # soft_meanshift
    num_iters: int = 10,
    temperature: float = 1.0,
    max_seeds: int = 256,
    # misc
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Cluster pixel / voxel embeddings into instance labels.

    Works identically on 2-D and 3-D inputs.  Not differentiable unless
    ``algorithm='soft_meanshift'``.

    Args:
        embedding: ``[E, *spatial]`` embedding tensor (unbatched).
        foreground_mask: Optional ``[*spatial]`` bool mask; background
            voxels always receive label 0.
        algorithm: One of ``{"meanshift", "hdbscan", "soft_meanshift",
            "spatial_cc"}``.  ``"spatial_cc"`` ignores ``max_points``,
            ``min_samples``, ``cluster_selection_epsilon`` and all
            SoftMeanShift knobs; it uses only ``bandwidth`` (= the
            embedding-distance threshold) and ``min_cluster_size``.
        bandwidth: Euclidean bandwidth for MeanShift / SoftMeanShift
            (also the edge threshold for ``spatial_cc``).  For
            discriminative-loss embeddings this should match ``delta_v``
            (= 0.5 in the original paper).
        min_cluster_size: Clusters with fewer than this many voxels are
            discarded (mapped to background).
        normalize_embeddings: L2-normalise embeddings before clustering.
            Must match the flag used at training time.
        backend: ``"auto"`` (cuML → CPU fallback), ``"cuml"``,
            ``"hdbscan"``, ``"sklearn"``, or ``"torch"``
            (``soft_meanshift`` only).
        min_samples: HDBSCAN ``min_samples`` (defaults to
            ``min_cluster_size`` if unset).
        cluster_selection_epsilon: HDBSCAN selection epsilon in embedding
            space; defaults to ``bandwidth`` so it matches the training
            margin.
        max_points: Upper bound on points passed to HDBSCAN; larger
            foreground sets are uniformly subsampled and the remaining
            points are assigned by nearest cluster center in embedding
            space.
        bin_seeding: MeanShift seed-grid flag.
        num_iters / temperature / max_seeds: SoftMeanShift knobs.
        seed: RNG seed for subsampling reproducibility.

    Returns:
        ``[*spatial]`` ``torch.long`` label tensor (0 = background,
        1..K = instances) on the same device as ``embedding``.
    """
    if algorithm not in _VALID_ALGOS:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. Choose from {_VALID_ALGOS}."
        )
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from {_VALID_BACKENDS}."
        )

    if algorithm == "soft_meanshift":
        return _cluster_soft_meanshift(
            embedding=embedding,
            foreground_mask=foreground_mask,
            bandwidth=bandwidth,
            num_iters=num_iters,
            temperature=temperature,
            min_cluster_size=min_cluster_size,
            normalize_embeddings=normalize_embeddings,
            max_seeds=max_seeds,
        )

    if algorithm == "spatial_cc":
        # Spatial-affinity CC uses bandwidth as the embedding-distance
        # threshold (semantically identical to delta_v).  max_points,
        # min_samples, cluster_selection_epsilon are all irrelevant here.
        cc_backend = backend if backend in ("auto", "cupy", "scipy") else "auto"
        return cluster_spatial_cc(
            embedding,
            foreground_mask=foreground_mask,
            delta_v=bandwidth,
            min_cluster_size=min_cluster_size,
            normalize_embeddings=normalize_embeddings,
            backend=cc_backend,
        )

    device = embedding.device
    emb_fg_np, fg_idx, spatial_shape = _as_fg_np(
        embedding, foreground_mask, normalize_embeddings,
    )
    labels_full = torch.zeros(
        int(np.prod(spatial_shape)), device=device, dtype=torch.long,
    )
    if len(fg_idx) == 0:
        return _reshape_to_spatial(labels_full, spatial_shape)

    eps = (
        float(cluster_selection_epsilon)
        if cluster_selection_epsilon is not None
        else float(bandwidth)
    )

    rng = np.random.default_rng(seed)

    if algorithm == "meanshift":
        # MeanShift scales ~O(N²); subsample like HDBSCAN for large volumes.
        sub_emb, sub_idx = _subsample(emb_fg_np, max_points, rng)
        sub_labels = _run_meanshift(
            sub_emb, bandwidth=bandwidth, bin_seeding=bin_seeding, backend=backend,
        )
    elif algorithm == "hdbscan":
        sub_emb, sub_idx = _subsample(emb_fg_np, max_points, rng)
        sub_labels = _run_hdbscan(
            sub_emb,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=eps,
            backend=backend,
        )
    else:  # pragma: no cover -- guarded above
        raise AssertionError(algorithm)

    if sub_idx is None:
        # No subsampling: labels correspond 1-1 to foreground rows.
        # Shift to 1-indexed (noise -1 -> 0) and remap to contiguous ids.
        raw = np.where(sub_labels >= 0, sub_labels + 1, 0).astype(np.int64)
    else:
        raw = _propagate_labels(
            emb_fg=emb_fg_np,
            sub_idx=sub_idx,
            sub_labels=sub_labels,
            epsilon=(2.0 * eps) if eps > 0 else float("inf"),
        )
    fg_labels = _remap_consecutive(raw, min_cluster_size)

    fg_labels_t = torch.from_numpy(fg_labels).to(device=device, dtype=torch.long)
    labels_full[fg_idx] = fg_labels_t
    return _reshape_to_spatial(labels_full, spatial_shape)


# ---------------------------------------------------------------------------
# Soft mean-shift dispatcher (differentiable, torch-native)
# ---------------------------------------------------------------------------

def _cluster_soft_meanshift(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor],
    bandwidth: float,
    num_iters: int,
    temperature: float,
    min_cluster_size: int,
    normalize_embeddings: bool,
    max_seeds: int,
) -> torch.Tensor:
    """Call :class:`SoftMeanShift` and return labels only.

    ``cluster_embeddings`` operates on unbatched ``[E, *spatial]`` tensors,
    matching the MeanShift / HDBSCAN paths.  For batched processing
    instantiate ``SoftMeanShift`` directly — it already supports
    ``[B, E, *spatial]``.
    """
    from neurons.inference.clusterer import SoftMeanShift

    embedding = rearrange(embedding, "... -> 1 ...")
    if foreground_mask is not None:
        foreground_mask = rearrange(foreground_mask, "... -> 1 ...")

    clusterer = SoftMeanShift(
        bandwidth=bandwidth,
        num_iters=num_iters,
        temperature=temperature,
        min_cluster_size=min_cluster_size,
        normalize_embeddings=normalize_embeddings,
    )
    labels, _, _ = clusterer(embedding, foreground_mask, max_seeds=max_seeds)
    return rearrange(labels, "1 ... -> ...")


# ---------------------------------------------------------------------------
# Backward-compatible wrappers
# ---------------------------------------------------------------------------

def cluster_embeddings_meanshift(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bandwidth: float = 0.5,
    min_cluster_size: int = 50,
    normalize_embeddings: bool = False,
    backend: str = "auto",
    max_points: int = 200_000,
) -> torch.Tensor:
    """Cluster pixel embeddings via MeanShift (cuML GPU → sklearn CPU)."""
    return cluster_embeddings(
        embedding,
        foreground_mask=foreground_mask,
        algorithm="meanshift",
        bandwidth=bandwidth,
        min_cluster_size=min_cluster_size,
        normalize_embeddings=normalize_embeddings,
        backend=backend,
        max_points=max_points,
    )


def cluster_embeddings_hdbscan(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bandwidth: float = 0.5,
    min_cluster_size: int = 50,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: Optional[float] = None,
    normalize_embeddings: bool = False,
    backend: str = "auto",
    max_points: int = 200_000,
) -> torch.Tensor:
    """Cluster pixel embeddings via HDBSCAN (cuML GPU → hdbscan → sklearn)."""
    return cluster_embeddings(
        embedding,
        foreground_mask=foreground_mask,
        algorithm="hdbscan",
        bandwidth=bandwidth,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        normalize_embeddings=normalize_embeddings,
        backend=backend,
        max_points=max_points,
    )


def cluster_embeddings_soft(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bandwidth: float = 0.5,
    num_iters: int = 10,
    temperature: float = 1.0,
    min_cluster_size: int = 50,
    normalize_embeddings: bool = False,
) -> torch.Tensor:
    """Differentiable soft mean-shift; preserves gradients through clustering."""
    return cluster_embeddings(
        embedding,
        foreground_mask=foreground_mask,
        algorithm="soft_meanshift",
        bandwidth=bandwidth,
        num_iters=num_iters,
        temperature=temperature,
        min_cluster_size=min_cluster_size,
        normalize_embeddings=normalize_embeddings,
    )


def cluster_embeddings_spatial_cc(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    delta_v: float = 0.5,
    min_cluster_size: int = 50,
    connectivity: int = 1,
    normalize_embeddings: bool = False,
    backend: str = "auto",
) -> torch.Tensor:
    """Connected components on the spatial-neighbour embedding-affinity graph."""
    return cluster_spatial_cc(
        embedding,
        foreground_mask=foreground_mask,
        delta_v=delta_v,
        min_cluster_size=min_cluster_size,
        connectivity=connectivity,
        normalize_embeddings=normalize_embeddings,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Offset-based (Hough voting) -- unchanged
# ---------------------------------------------------------------------------

def cluster_offsets_hough(
    offsets: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bin_size: float = 2.0,
    sigma: float = 2.0,
    threshold: float = 0.3,
    min_votes: int = 50,
) -> torch.Tensor:
    """Cluster via Hough voting on predicted spatial offsets."""
    from neurons.inference.clusterer import HoughVoting

    batched = offsets.dim() >= 4
    if not batched:
        offsets = rearrange(offsets, "... -> 1 ...")
        if foreground_mask is not None:
            foreground_mask = rearrange(foreground_mask, "... -> 1 ...")

    voter = HoughVoting(
        bin_size=bin_size,
        sigma=sigma,
        threshold=threshold,
        min_votes=min_votes,
    )
    labels = voter(offsets, foreground_mask)

    if not batched:
        labels = rearrange(labels, "1 ... -> ...")
    return labels


# ---------------------------------------------------------------------------
# Introspection helper (useful for logging which backend ran)
# ---------------------------------------------------------------------------

def available_backends() -> Dict[str, Dict[str, bool]]:
    """Report which clustering backends are installed on this machine."""
    return {
        "meanshift": {
            "cuml": _probe_cuml_meanshift() is not None,
            "sklearn": _probe_sklearn_meanshift() is not None,
        },
        "hdbscan": {
            "cuml": _probe_cuml_hdbscan() is not None,
            "hdbscan": _probe_hdbscan_pkg() is not None,
            "sklearn": _probe_sklearn_hdbscan() is not None,
        },
        "soft_meanshift": {"torch": True},
        "spatial_cc": {
            "cupy": _probe_cupy_csgraph() is not None,
            "scipy": _probe_scipy_csgraph() is not None,
        },
    }
