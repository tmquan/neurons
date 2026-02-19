#!/usr/bin/env python
"""
Vista3D Inference Script with Sliding Window and Agglomeration.

Supports:
- Automatic segmentation of full volumes
- Gaussian / average / max aggregation for overlap blending
- Per-class instance segmentation via --class-id
- Dual-head (semantic + instance) or single-head models

Usage:
    # Automatic segmentation (all classes)
    python scripts/infer_vista3d.py \
        --checkpoint checkpoints/vista3d/best.ckpt \
        --input data/SNEMI3D/AC3_inputs.h5 \
        --output output/AC3_segmentation.h5

    # Segment only neurons (class 1)
    python scripts/infer_vista3d.py \
        --checkpoint checkpoints/vista3d/best.ckpt \
        --input data/volume.h5 \
        --output output/neurons.h5 \
        --class-id 1

    # Custom sliding window settings
    python scripts/infer_vista3d.py \
        --checkpoint checkpoints/vista3d/best.ckpt \
        --input data/volume.h5 \
        --output output/segmentation.h5 \
        --patch-size 128 128 128 \
        --stride 64 64 64 \
        --aggregation gaussian
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from einops import rearrange

from neurons.inference.sliding_window import sliding_window_inference
from neurons.inference.soft_clustering import SoftMeanShift
from neurons.inference.stitcher import EmbeddingStitcher
from neurons.utils.io import load_volume, save_volume


def _normalize(volume: torch.Tensor) -> torch.Tensor:
    vmin, vmax = volume.min(), volume.max()
    return (volume - vmin) / (vmax - vmin + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Vista3D Inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input volume")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output segmentation")

    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128],
                        help="Patch size (D H W)")
    parser.add_argument("--stride", type=int, nargs=3, default=None,
                        help="Stride between patches. Default: patch_size // 2")
    parser.add_argument("--aggregation", type=str, default="gaussian",
                        choices=["gaussian", "average", "max"],
                        help="Aggregation method for overlapping patches")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for patch processing")

    parser.add_argument("--class-id", type=int, default=None,
                        help="Restrict instance segmentation to this semantic class")
    parser.add_argument("--bandwidth", type=float, default=0.5,
                        help="Mean-shift bandwidth for instance clustering")
    parser.add_argument("--merge-threshold", type=float, default=0.5,
                        help="Embedding distance threshold for cross-patch merging")
    parser.add_argument("--min-instance-size", type=int, default=50,
                        help="Minimum voxels per instance")

    parser.add_argument("--input-key", type=str, default=None,
                        help="Key for HDF5/NPZ input files")
    parser.add_argument("--output-key", type=str, default="segmentation",
                        help="Key for HDF5 output")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--output-probs", action="store_true",
                        help="Output semantic probabilities instead of instance labels")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.checkpoint}")
    from neurons.modules.vista3d_module import Vista3DModule
    module = Vista3DModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model = module.model.to(device)
    model.eval()

    print(f"Loading input: {args.input}")
    load_kw = {"key": args.input_key} if args.input_key else {}
    volume_np = load_volume(args.input, **load_kw)
    print(f"Input shape: {volume_np.shape}, dtype: {volume_np.dtype}")

    volume_tensor = torch.from_numpy(volume_np.astype(np.float32))
    volume_tensor = _normalize(volume_tensor)

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride) if args.stride else None

    print(f"\nRunning sliding window inference...")
    print(f"  Patch size: {patch_size}")
    print(f"  Stride: {stride or 'auto (50% overlap)'}")
    print(f"  Aggregation: {args.aggregation}")

    result = sliding_window_inference(
        model,
        volume_tensor,
        patch_size=patch_size,
        stride=stride,
        aggregation=args.aggregation,
        batch_size=args.batch_size,
        device=device,
        progress=True,
    )

    is_dual = isinstance(result, dict)

    if args.output_probs:
        sem_probs = result["semantic_probs"] if is_dual else result
        output = sem_probs.cpu().numpy()
        print(f"Output probabilities shape: {output.shape}")
        save_volume(output, args.output, key=args.output_key)
        print("Done.")
        return

    if is_dual:
        sem_probs = result["semantic_probs"]
        emb = result["instance_embeddings"]

        class_ids = sem_probs.argmax(dim=0)
        print(f"Semantic classes detected: {torch.unique(class_ids).tolist()}")

        if args.class_id is not None:
            classes_to_process = [args.class_id]
        else:
            all_classes = torch.unique(class_ids).tolist()
            classes_to_process = [c for c in all_classes if c > 0]

        clusterer = SoftMeanShift(
            bandwidth=args.bandwidth,
            min_cluster_size=args.min_instance_size,
        )

        vol_shape = class_ids.shape
        final_labels = torch.zeros(vol_shape, device=device, dtype=torch.long)
        next_id = 1

        for cid in classes_to_process:
            print(f"  Clustering class {cid}...")
            mask = class_ids == cid
            if mask.sum() == 0:
                continue

            emb_batched = rearrange(emb, "e ... -> 1 e ...")
            mask_batched = rearrange(mask, "... -> 1 ...")
            labels_c, _, _ = clusterer(emb_batched, mask_batched)
            labels_c = rearrange(labels_c, "1 ... -> ...")

            for uid in torch.unique(labels_c):
                if uid == 0:
                    continue
                final_labels[labels_c == uid] = next_id
                next_id += 1

        stitcher = EmbeddingStitcher(
            merge_threshold=args.merge_threshold,
            min_instance_size=args.min_instance_size,
        )

        stride_used = stride or (patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2)
        D, H, W = vol_shape
        positions = []
        for i in range(max(1, (D - patch_size[0] + stride_used[0]) // stride_used[0])):
            for j in range(max(1, (H - patch_size[1] + stride_used[1]) // stride_used[1])):
                for k in range(max(1, (W - patch_size[2] + stride_used[2]) // stride_used[2])):
                    positions.append((
                        min(i * stride_used[0], D - patch_size[0]),
                        min(j * stride_used[1], H - patch_size[1]),
                        min(k * stride_used[2], W - patch_size[2]),
                    ))

        final_labels = stitcher.stitch(final_labels, emb, positions, patch_size)
        output = final_labels.cpu().numpy().astype(np.uint32)

    else:
        output = result.argmax(dim=0).cpu().numpy().astype(np.uint16)

    print(f"\nSaving output to: {args.output}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    num_instances = len(np.unique(output)) - (1 if 0 in output else 0)
    print(f"Instances found: {num_instances}")
    save_volume(output, args.output, key=args.output_key)
    print("Inference complete!")


if __name__ == "__main__":
    main()
