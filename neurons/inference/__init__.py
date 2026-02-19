"""
Inference utilities for connectomics segmentation.

Includes:
- SoftMeanShift: Differentiable mean-shift clustering for embeddings
- HoughVoting: Differentiable Hough voting for offset-based embeddings
- create_gaussian_weight: 3D Gaussian weight map for sliding window blending
- sliding_window_inference: Gaussian-weighted sliding window for full volumes
- EmbeddingStitcher: Merge-split reconciliation of instance IDs across patches
"""

from neurons.inference.soft_clustering import SoftMeanShift, HoughVoting
from neurons.inference.sliding_window import (
    create_gaussian_weight,
    sliding_window_inference,
)
from neurons.inference.stitcher import EmbeddingStitcher

__all__ = [
    "SoftMeanShift",
    "HoughVoting",
    "create_gaussian_weight",
    "sliding_window_inference",
    "EmbeddingStitcher",
]
