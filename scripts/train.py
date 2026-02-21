#!/usr/bin/env python
"""
Neurons Training Script

Main entry point for training connectomics segmentation models.
Uses Hydra for configuration management.

Usage:
    # Train with default config
    python scripts/train.py

    # Train with specific dataset config
    python scripts/train.py --config-name snemi3d

    # Override parameters via CLI
    python scripts/train.py --config-name snemi3d \\
        data.batch_size=8 training.max_epochs=200

    # Fast development run
    python scripts/train.py training.fast_dev_run=true
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Enable Tensor Core optimization
torch.set_float32_matmul_precision("high")


def get_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Create appropriate datamodule based on config."""
    from neurons.datamodules import (
        CREMI3DDataModule,
        CombineDataModule,
        MICRONSDataModule,
        MitoEM2DataModule,
        SNEMI3DDataModule,
    )

    data_cfg = cfg.data
    dataset_type = data_cfg.get("dataset", "snemi3d").lower()

    common_args: Dict[str, Any] = {
        "data_root": data_cfg.get("data_root", "data"),
        "batch_size": data_cfg.get("batch_size", 4),
        "num_workers": data_cfg.get("num_workers", 4),
        "train_val_split": data_cfg.get("train_val_split", 0.2),
        "cache_rate": data_cfg.get("cache_rate", 0.5),
        "pin_memory": data_cfg.get("pin_memory", True),
    }

    image_size = data_cfg.get("image_size")
    if image_size is not None:
        common_args["image_size"] = tuple(image_size) if isinstance(image_size, list) else image_size

    if dataset_type == "snemi3d":
        patch_size = data_cfg.get("patch_size")
        return SNEMI3DDataModule(
            slice_mode=data_cfg.get("slice_mode", True),
            num_samples=data_cfg.get("num_samples"),
            patch_size=tuple(patch_size) if patch_size else None,
            **common_args,
        )

    elif dataset_type == "cremi3d":
        patch_size = data_cfg.get("patch_size")
        volumes = data_cfg.get("volumes")
        return CREMI3DDataModule(
            volumes=list(volumes) if volumes else None,
            include_clefts=data_cfg.get("include_clefts", True),
            include_mito=data_cfg.get("include_mito", False),
            num_samples=data_cfg.get("num_samples"),
            patch_size=tuple(patch_size) if patch_size else None,
            **common_args,
        )

    elif dataset_type == "microns":
        patch_size = data_cfg.get("patch_size")
        return MICRONSDataModule(
            volume_file=data_cfg.get("volume_file", "volume"),
            segmentation_file=data_cfg.get("segmentation_file", "segmentation"),
            include_synapses=data_cfg.get("include_synapses", False),
            include_mitochondria=data_cfg.get("include_mitochondria", False),
            slice_mode=data_cfg.get("slice_mode", True),
            num_samples=data_cfg.get("num_samples"),
            patch_size=tuple(patch_size) if patch_size else None,
            **common_args,
        )

    elif dataset_type == "mitoem2":
        patch_size = data_cfg.get("patch_size")
        return MitoEM2DataModule(
            split=data_cfg.get("split", "human"),
            slice_mode=data_cfg.get("slice_mode", True),
            num_samples=data_cfg.get("num_samples"),
            patch_size=tuple(patch_size) if patch_size else None,
            **common_args,
        )

    elif dataset_type == "combine":
        datasets_cfg = data_cfg.get("datasets", {})
        snemi3d_cfg = datasets_cfg.get("snemi3d", {})
        cremi3d_cfg = datasets_cfg.get("cremi3d", {})
        mitoem2_cfg = datasets_cfg.get("mitoem2", {})

        patch_size = data_cfg.get("patch_size", [32, 128, 128])
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)

        dm_entries: Dict[str, tuple] = {}

        if snemi3d_cfg.get("enabled", True):
            snemi3d_root = snemi3d_cfg.get("data_root", "data/snemi3d")
            if Path(snemi3d_root).exists():
                dm_entries["snemi3d"] = (
                    SNEMI3DDataModule(
                        data_root=snemi3d_root,
                        batch_size=data_cfg.get("batch_size", 4),
                        num_workers=data_cfg.get("num_workers", 4),
                        train_val_split=data_cfg.get("train_val_split", 0.2),
                        cache_rate=data_cfg.get("cache_rate", 0.5),
                        patch_size=patch_size,
                        slice_mode=False,
                    ),
                    snemi3d_cfg.get("weight", 1.0),
                )

        if cremi3d_cfg.get("enabled", True):
            cremi3d_root = cremi3d_cfg.get("data_root", "data/cremi3d")
            if Path(cremi3d_root).exists():
                dm_entries["cremi3d"] = (
                    CREMI3DDataModule(
                        data_root=cremi3d_root,
                        batch_size=data_cfg.get("batch_size", 4),
                        num_workers=data_cfg.get("num_workers", 4),
                        train_val_split=data_cfg.get("train_val_split", 0.2),
                        cache_rate=data_cfg.get("cache_rate", 0.5),
                        patch_size=patch_size,
                        volumes=list(cremi3d_cfg.get("volumes", ["A", "B"])),
                        include_clefts=cremi3d_cfg.get("include_clefts", True),
                        include_mito=cremi3d_cfg.get("include_mito", False),
                    ),
                    cremi3d_cfg.get("weight", 1.0),
                )

        if mitoem2_cfg.get("enabled", False):
            mito_root = mitoem2_cfg.get("data_root", "data/mitoem2")
            if Path(mito_root).exists():
                dm_entries["mitoem2"] = (
                    MitoEM2DataModule(
                        data_root=mito_root,
                        batch_size=data_cfg.get("batch_size", 4),
                        num_workers=data_cfg.get("num_workers", 4),
                        train_val_split=data_cfg.get("train_val_split", 0.2),
                        cache_rate=data_cfg.get("cache_rate", 0.5),
                        split=mitoem2_cfg.get("split", "human"),
                        slice_mode=False,
                    ),
                    mitoem2_cfg.get("weight", 1.5),
                )

        return CombineDataModule(
            datamodules=dm_entries if dm_entries else None,
            batch_size=data_cfg.get("batch_size", 4),
            num_workers=data_cfg.get("num_workers", 4),
            use_weighted_sampling=True,
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_module(cfg: DictConfig) -> pl.LightningModule:
    """Create appropriate Lightning module based on config."""
    from neurons.modules import Vista3DModule, Vista2DModule

    model_cfg = dict(cfg.get("model", {}))
    optimizer_cfg = dict(cfg.get("optimizer", {}))
    loss_cfg = dict(cfg.get("loss", {}))
    training_cfg = dict(cfg.get("training", {}))

    model_type = model_cfg.pop("type", "vista3d").lower()

    if model_type == "vista2d":
        return Vista2DModule(
            model_config=model_cfg,
            optimizer_config=optimizer_cfg,
            loss_config=loss_cfg,
            training_config=training_cfg,
        )

    return Vista3DModule(
        model_config=model_cfg,
        optimizer_config=optimizer_cfg,
        loss_config=loss_cfg,
        training_config=training_cfg,
    )


def setup_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """Setup training callbacks from configuration."""
    output_dir = cfg.get("output_dir", "outputs")
    callbacks: List[pl.Callback] = []

    callback_cfg = cfg.get("callbacks", {})

    ckpt_cfg = callback_cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled", True):
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_cfg.get("dirpath") or str(Path(output_dir) / "checkpoints"),
                filename=ckpt_cfg.get("filename", "{epoch:02d}-{val/loss:.4f}"),
                save_top_k=ckpt_cfg.get("save_top_k", 3),
                monitor=ckpt_cfg.get("monitor", "val/loss"),
                mode=ckpt_cfg.get("mode", "min"),
                save_last=ckpt_cfg.get("save_last", True),
                verbose=ckpt_cfg.get("verbose", True),
                auto_insert_metric_name=False,
            )
        )

    es_cfg = callback_cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val/loss"),
                patience=es_cfg.get("patience", 20),
                mode=es_cfg.get("mode", "min"),
                verbose=es_cfg.get("verbose", True),
                min_delta=es_cfg.get("min_delta", 0.0),
            )
        )

    if callback_cfg.get("lr_monitor", {}).get("enabled", True):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    img_cfg = callback_cfg.get("image_logger", {})
    if img_cfg.get("enabled", False):
        from neurons.callbacks import ImageLogger
        spatial = cfg.get("model", {}).get("type", "vista2d")
        callbacks.append(
            ImageLogger(
                every_n_epochs=img_cfg.get("every_n_epochs", 1),
                max_images=img_cfg.get("max_images", 4),
                spatial_dims=3 if "3d" in spatial else 2,
            )
        )

    callbacks.append(RichProgressBar())
    callbacks.append(ModelSummary(max_depth=2))

    return callbacks


def setup_logger(cfg: DictConfig) -> Any:
    """Setup experiment logger."""
    output_dir = cfg.get("output_dir", "outputs")
    logger_type = cfg.get("logger", "tensorboard")

    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=str(Path(output_dir) / "logs"),
            name=cfg.get("experiment_name", "neurons"),
            version=None,
        )
    elif logger_type == "wandb":
        return WandbLogger(
            project=cfg.get("project_name", "neurons"),
            name=f"{cfg.get('experiment_name', 'run')}_{cfg.get('seed', 42)}",
            save_dir=str(Path(output_dir) / "logs"),
        )
    else:
        return True


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    print("=" * 60)
    print("Neurons - Connectomics Segmentation Training")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    print(f"\nRandom seed: {seed}")

    datamodule = get_datamodule(cfg)
    print(f"\nDataModule: {datamodule.__class__.__name__}")
    print(f"  Dataset: {cfg.data.get('dataset', 'snemi3d')}")
    print(f"  Data root: {cfg.data.get('data_root', 'data')}")
    print(f"  Batch size: {cfg.data.get('batch_size', 4)}")

    module = get_module(cfg)
    print(f"\nModule: {module.__class__.__name__}")

    callbacks = setup_callbacks(cfg)
    print(f"\nCallbacks: {len(callbacks)} registered")

    logger = setup_logger(cfg)
    print(f"Logger: {cfg.get('logger', 'tensorboard')}")

    training_cfg = cfg.training
    trainer = pl.Trainer(
        max_epochs=training_cfg.get("max_epochs", 100),
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", 1),
        strategy=training_cfg.get("strategy", "auto"),
        precision=training_cfg.get("precision", "32-true"),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=training_cfg.get("log_every_n_steps", 50),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        val_check_interval=training_cfg.get("val_check_interval", 1.0),
        check_val_every_n_epoch=training_cfg.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=training_cfg.get("num_sanity_val_steps", 2),
        enable_progress_bar=training_cfg.get("enable_progress_bar", True),
        enable_model_summary=training_cfg.get("enable_model_summary", True),
        deterministic=training_cfg.get("deterministic", False),
        benchmark=training_cfg.get("benchmark", True),
        fast_dev_run=training_cfg.get("fast_dev_run", False),
    )

    print(f"\nTrainer initialized:")
    print(f"  Max epochs: {training_cfg.get('max_epochs', 100)}")
    print(f"  Accelerator: {training_cfg.get('accelerator', 'auto')}")
    print(f"  Devices: {training_cfg.get('devices', 1)}")
    print(f"  Precision: {training_cfg.get('precision', '32-true')}")

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    try:
        trainer.fit(module, datamodule)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        raise

    if trainer.global_rank == 0:
        output_dir = cfg.get("output_dir", "outputs")
        final_path = Path(output_dir) / "checkpoints" / "final_model.ckpt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(final_path))
        print(f"\nFinal model saved: {final_path}")

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
