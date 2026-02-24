"""Persistent process pool for parallelizing per-instance CPU-bound operations.

Uses ``torch.multiprocessing`` with ``fork`` to share memory.  Worker
functions receive and return numpy arrays to avoid pickling torch tensors.

Usage::

    from neurons.utils.parallel import pmap

    # Process each instance mask in parallel
    results = pmap(worker_fn, list_of_args)
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Optional

import torch.multiprocessing as mp

_POOL: Optional[ProcessPoolExecutor] = None
_POOL_SIZE: int = 0


def _get_pool(n_workers: Optional[int] = None) -> ProcessPoolExecutor:
    """Get or create a persistent process pool."""
    global _POOL, _POOL_SIZE

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    if _POOL is None or _POOL_SIZE != n_workers:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
        ctx = mp.get_context("fork")
        _POOL = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
        _POOL_SIZE = n_workers

    return _POOL


def pmap(
    fn: Callable,
    args_list: List[Any],
    n_workers: Optional[int] = None,
) -> List[Any]:
    """Parallel map over a list of arguments using a persistent process pool.

    Falls back to sequential execution if the list is small or
    multiprocessing is unavailable.

    Args:
        fn: Worker function. Must accept a single argument (tuple) and
            return a result. Should operate on numpy arrays, not torch tensors.
        args_list: List of argument tuples, one per work item.
        n_workers: Number of worker processes (default: min(cpu_count, 8)).

    Returns:
        List of results in the same order as args_list.
    """
    if len(args_list) == 0:
        return []

    if len(args_list) <= 2:
        return [fn(a) for a in args_list]

    try:
        pool = _get_pool(n_workers)
        futures = [pool.submit(fn, a) for a in args_list]
        return [f.result() for f in futures]
    except Exception:
        return [fn(a) for a in args_list]
