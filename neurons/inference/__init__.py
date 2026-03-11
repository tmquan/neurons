"""
Inference utilities for connectomics segmentation.

Includes:
- SoftMeanShift: Differentiable mean-shift clustering for embeddings
- HoughVoting: Differentiable Hough voting for offset-based embeddings
- create_gaussian_weight: 3D Gaussian weight map for sliding window blending
- sliding_window_inference: Gaussian-weighted sliding window for full volumes
- EmbeddingStitcher: Merge-split reconciliation of instance IDs across patches
"""

# TODO: Implement inference utilities