"""Training CLI entry point for Jaeger (PyTorch).

All core model-building logic lives in `jaeger.nnlib.pytorch.builder`.
This module is a thin Click wrapper that orchestrates:
- strategy selection (GPU/CPU)
- data pipeline construction
- model training loops
- saving
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# temporary fix
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"

import click
import torch
import torch.nn as nn

from jaeger.dataops.pytorch.builders import build_datasets
from jaeger.nnlib.pytorch.builder import ModelBuilder
from jaeger.training.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
)
from jaeger.training.pytorch.distributed import (
    cleanup_distributed,
    get_device,
    setup_distributed,
)
from jaeger.training.pytorch.trainer import Trainer
from jaeger.utils.logging import get_logger
from jaeger.utils.misc import load_model_config, numerize

try:
    from icecream import ic

    ic.configureOutput(prefix="Jaeger |")
except ImportError:

    def ic(*args, **kwargs):
        pass


logger = get_logger(log_file=None, log_path=None, level=3)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) for *model*."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _resolve_steps(value: Optional[int]) -> Optional[int]:
    """Convert a config step count to a value understood by the engine.

    ``None`` or negative values mean "run until the dataloader is exhausted".
    Zero or positive values limit the loop to that many batches.
    """
    if value is None or value < 0:
        return None
    return int(value)


def _initialize_lazy_layers(
    models: Dict[str, nn.Module], config: Dict[str, Any]
) -> None:
    """Run a dummy forward pass to materialize any lazy layers."""
    model_cfg = config.get("model", {})
    embedding_cfg = model_cfg.get("embedding", {})
    sp_cfg = model_cfg.get("string_processor", {})
    input_type = embedding_cfg.get("input_type", "translated")
    input_shape = embedding_cfg.get("input_shape")
    crop_size = int(sp_cfg.get("crop_size", 500))
    device = next(models["jaeger_classifier"].parameters()).device

    # Use a short dummy length; lazy layers only need to infer channel dims.
    length = 64 if input_type == "nucleotide" else max(1, crop_size // 3 - 1)

    if input_shape is not None and len(input_shape) == 3:
        # One-hot input, e.g. [2, null, 4] or [6, null, vocab_size].
        frames = int(input_shape[0])
        channels = int(input_shape[2])
        dummy = torch.zeros((1, frames, length, channels), dtype=torch.float32, device=device)
    else:
        # Integer-index input, e.g. [2, null] or [6, null].
        frames = int(input_shape[0]) if input_shape else (
            2 if input_type == "nucleotide" else 6
        )
        dummy = torch.zeros((1, frames, length), dtype=torch.long, device=device)

    mask = torch.ones(
        dummy.shape[:-1] if dummy.dim() == 4 else dummy.shape,
        dtype=torch.bool,
        device=device,
    )
    with torch.no_grad():
        models["jaeger_classifier"](dummy, mask)


def _load_checkpoint_if_requested(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Optional[str | Path],
    device: torch.device,
) -> int:
    """Load model and optimizer state dicts from *checkpoint_path* if provided.

    The checkpoint is mapped to *device* so that both model parameters and
    optimizer state land on the target device and stay in sync.

    Returns the epoch stored in the checkpoint, or ``0`` if no checkpoint was
    loaded or no epoch was recorded.
    """
    if checkpoint_path is None:
        return 0
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning("Checkpoint path does not exist: %s", path)
        return 0
    logger.info("Loading checkpoint from %s", path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))


def _normalize_optimizer_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert TF-style optimizer params to PyTorch-compatible ones."""
    if not params:
        return {}
    normalized = dict(params)
    if "learning_rate" in normalized:
        normalized["lr"] = normalized.pop("learning_rate")
    for key in ("clipnorm", "clipvalue", "global_clipnorm"):
        if key in normalized:
            logger.warning("Ignoring unsupported optimizer parameter: %s", key)
            normalized.pop(key)
    return normalized


def _callback_path_from_config(
    callbacks_cfg: List[Dict[str, Any]], name: str
) -> Optional[Dict[str, Any]]:
    """Return the first callback entry matching *name* from the callbacks list."""
    for entry in callbacks_cfg or []:
        if entry.get("name") == name:
            return entry.get("params", {}) or {}
    return None


_SUPPORTED_CALLBACKS = {"EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau", "TerminateOnNaN"}


def _build_callbacks(train_cfg: Dict[str, Any]) -> List[Any]:
    """Build PyTorch callbacks from the training config."""
    callbacks: List[Any] = []
    callbacks_cfg = train_cfg.get("callbacks", {}).get("classifier", [])

    for entry in callbacks_cfg:
        name = entry.get("name")
        params = entry.get("params") or {}
        if name == "EarlyStopping":
            patience = params.get("patience") or train_cfg.get("patience")
            if patience is not None:
                callbacks.append(
                    EarlyStopping(
                        monitor=params.get("monitor", "val_loss"),
                        patience=int(patience),
                        mode=params.get("mode", "min"),
                        restore_best_weights=params.get("restore_best_weights", True),
                    )
                )
        elif name == "ModelCheckpoint":
            checkpoint_path = params.get("filepath") or train_cfg.get("checkpoint_path")
            if checkpoint_path is not None:
                callbacks.append(
                    ModelCheckpoint(
                        filepath=checkpoint_path,
                        monitor=params.get("monitor", "val_loss"),
                        mode=params.get("mode", "min"),
                        save_best_only=params.get("save_best_only", True),
                        verbose=int(params.get("verbose", 1) or 0),
                    )
                )
        elif name == "ReduceLROnPlateau":
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=params.get("monitor", "val_loss"),
                    mode=params.get("mode", "min"),
                    patience=int(params.get("patience", 2)),
                    factor=float(params.get("factor", 0.5)),
                    min_lr=float(params.get("min_lr", 1e-6)),
                    verbose=int(params.get("verbose", 0) or 0),
                )
            )
        elif name == "TerminateOnNaN":
            callbacks.append(
                TerminateOnNaN(monitor=params.get("monitor", "train_loss"))
            )
        elif name in _SUPPORTED_CALLBACKS:
            continue
        else:
            logger.warning("Ignoring unsupported callback: %s", name)

    return callbacks


def _find_latest_checkpoint(directory: Path) -> Optional[Path]:
    """Return the most recently modified ``.pt`` file in *directory*, if any."""
    if not directory.exists():
        return None
    checkpoints = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def _has_checkpoints(directory: Path) -> bool:
    """Return True if *directory* exists and contains any ``.pt`` files."""
    if not directory.exists():
        return False
    return any(directory.glob("*.pt"))


def _guard_against_overwrite(
    directory: Path,
    force: bool,
    from_last_checkpoint: bool,
    branch_name: str,
) -> None:
    """Raise an error if existing checkpoints would be overwritten.

    Existing checkpoints are allowed when resuming via ``from_last_checkpoint``
    or when explicitly overwriting via ``force``.
    """
    if force or from_last_checkpoint or not _has_checkpoints(directory):
        return
    raise click.ClickException(
        f"{branch_name} checkpoint directory already exists: {directory}. "
        "Use --force to overwrite or --from_last_checkpoint to resume training."
    )


def _save_model_checkpoint(
    model: nn.Module,
    save_dir: Path,
    filename: str,
    metadata: Optional[str] = None,
    num_params: Optional[str] = None,
) -> None:
    """Persist *model*'s state dict to ``save_dir / filename``."""
    save_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    if metadata is not None:
        payload["metadata"] = str(metadata)
    if num_params is not None:
        payload["num_params"] = num_params
    torch.save(payload, save_dir / filename)
    logger.info("Saved checkpoint to %s", save_dir / filename)


class _ReliabilityPipeline(nn.Module):
    """Runs the representation model and reliability head."""

    def __init__(self, rep_model: nn.Module, head: nn.Module):
        super().__init__()
        self.rep_model = rep_model
        self.head = head

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        outputs = self.rep_model(x, mask)
        if isinstance(outputs, tuple):
            nmd = outputs[1] if len(outputs) > 1 else outputs[0]
        else:
            nmd = outputs
        return self.head(nmd)


# ------------------------------------------------------------------
# CLI command
# ------------------------------------------------------------------


@click.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--mixed_precision", is_flag=True, default=False)
@click.option("--from_last_checkpoint", is_flag=True, default=False)
@click.option("--force", is_flag=True, default=False)
@click.option("--only_classification_head", is_flag=True, default=False)
@click.option("--only_reliability_head", is_flag=True, default=False)
@click.option("--only_heads", is_flag=True, default=False)
@click.option("--only_save", is_flag=True, default=False)
@click.option("--save_model", is_flag=True, default=False)
@click.option("--self_supervised_pretraining", is_flag=True, default=False)
@click.option(
    "--xla",
    is_flag=True,
    default=False,
    help="Enable XLA JIT compilation for training.",
)
@click.option("--meta", type=click.Path(), default=None)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help="Show per-section timing in the progress bar.",
)
def train_fragment(
    config,
    mixed_precision,
    from_last_checkpoint,
    force,
    only_classification_head,
    only_reliability_head,
    only_heads,
    only_save,
    save_model,
    self_supervised_pretraining,
    xla,
    meta,
    profile,
):
    """Train a fragment-level Jaeger model."""
    train_fragment_core(
        config=config,
        mixed_precision=mixed_precision,
        from_last_checkpoint=from_last_checkpoint,
        force=force,
        only_classification_head=only_classification_head,
        only_reliability_head=only_reliability_head,
        only_heads=only_heads,
        only_save=only_save,
        save_model=save_model,
        self_supervised_pretraining=self_supervised_pretraining,
        xla=xla,
        meta=meta,
        profile=profile,
    )


# ------------------------------------------------------------------
# Core training orchestration (CLI glue)
# ------------------------------------------------------------------


def train_fragment_core(**kwargs):
    """Train fragment classification and reliability models with PyTorch."""
    config = load_model_config(Path(kwargs["config"]))
    config["use_pytorch"] = True
    config["mix_precision"] = kwargs.get("mixed_precision", False)
    config["from_last_checkpoint"] = kwargs.get("from_last_checkpoint")
    config["force"] = kwargs.get("force")
    config["use_xla"] = kwargs.get("xla", False)

    if kwargs.get("mixed_precision", False):
        logger.warning(
            "mixed_precision is not implemented for PyTorch training yet; ignoring"
        )

    if kwargs.get("xla", False):
        logger.warning("XLA is not supported for PyTorch training; ignoring")

    if kwargs.get("self_supervised_pretraining", False):
        logger.warning("self_supervised_pretraining is not implemented yet; skipping")

    setup_distributed()
    device = get_device()
    logger.info("Using device: %s", device)

    try:
        builder = ModelBuilder(config)
        models = builder.build_fragment_classifier()
        rep_model = models["rep_model"]

        _initialize_lazy_layers(models, config)

        total_params, trainable_params = _count_parameters(rep_model)
        logger.info(
            "rep_model parameters: total=%s, trainable=%s",
            numerize(total_params, decimal=1),
            numerize(trainable_params, decimal=1),
        )
        model_num_params = numerize(total_params, decimal=1)

        # Move models to the target device before building optimizers so that
        # optimizer param_groups and any resumed optimizer state live on the
        # same device as the model.
        rep_model.to(device)
        models["jaeger_classifier"].to(device)
        if models.get("reliability_head") is not None:
            models["reliability_head"].to(device)

        train_cfg = config.get("training", {})
        classifier_dir = Path(train_cfg.get("classifier_dir", "checkpoints/classifier"))
        reliability_dir = Path(
            train_cfg.get("reliability_dir", "checkpoints/reliability")
        )
        model_save_cfg = train_cfg.get("model_saving", {}) or {}
        model_save_path = model_save_cfg.get("path")
        if model_save_path:
            model_save_path = Path(model_save_path)

        if kwargs.get("force", False):
            for directory in (classifier_dir, reliability_dir):
                if directory.exists():
                    shutil.rmtree(directory)
                    logger.info("Removed existing checkpoint directory: %s", directory)

        # ================= train classifier ======================
        if not kwargs.get("only_reliability_head", False):
            classifier_loaders = build_datasets(config, branch="classifier")

            # Normalize optimizer params before compiling so the builder uses them.
            opt_params = train_cfg.get("optimizer_params", {}) or {}
            train_cfg["optimizer_params"] = _normalize_optimizer_params(opt_params)

            class_weights = train_cfg.get("classifier_class_weights")
            if class_weights is not None:
                weight_tensor = torch.zeros(builder.classifier_out_dim)
                for idx, w in class_weights.items():
                    weight_tensor[int(idx)] = float(w)
                class_weights = weight_tensor.to(device)

            model, optimizer, loss_fn = builder.compile_model(
                models,
                train_branch="classifier",
                class_weights=class_weights,
            )

            checkpoint_path = None
            if kwargs.get("from_last_checkpoint", False):
                checkpoint_path = _find_latest_checkpoint(classifier_dir)
            start_epoch = _load_checkpoint_if_requested(
                model, optimizer, checkpoint_path, device
            )

            if kwargs.get("only_classification_head", False) or kwargs.get(
                "only_heads", False
            ):
                for param in rep_model.parameters():
                    param.requires_grad = False

            callbacks = _build_callbacks(train_cfg)
            epochs = int(train_cfg.get("classifier_epochs", 1))
            train_steps = _resolve_steps(train_cfg.get("classifier_train_steps"))
            validation_steps = _resolve_steps(
                train_cfg.get("classifier_validation_steps")
            )

            if kwargs.get("only_save", False):
                logger.info("Skipping classifier training (--only_save)")
            else:
                _guard_against_overwrite(
                    classifier_dir,
                    force=kwargs.get("force", False),
                    from_last_checkpoint=kwargs.get("from_last_checkpoint", False),
                    branch_name="Classifier",
                )
                trainer = Trainer(
                    model=model,
                    train_loader=classifier_loaders["train"],
                    val_loader=classifier_loaders["validation"],
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=epochs,
                    device=device,
                    metrics=builder.get_metrics(branch="classifier"),
                    callbacks=callbacks,
                    history_path=str(classifier_dir / "history.json"),
                    branch="classifier",
                    progress_bar=kwargs.get("progress_bar", False),
                    profile=kwargs.get("profile", False),
                    train_steps=train_steps,
                    validation_steps=validation_steps,
                    start_epoch=start_epoch,
                )
                trainer.fit()

        # ================= train reliability ======================
        if (
            "reliability_model" in config.get("model", {})
            and not kwargs.get("only_classification_head", False)
            and models.get("reliability_head") is not None
        ):
            try:
                reliability_loaders = build_datasets(config, branch="reliability")
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Reliability data not configured; skipping reliability training: %s",
                    exc,
                )
                reliability_loaders = None

            if reliability_loaders is not None and not kwargs.get("only_save", False):
                _guard_against_overwrite(
                    reliability_dir,
                    force=kwargs.get("force", False),
                    from_last_checkpoint=kwargs.get("from_last_checkpoint", False),
                    branch_name="Reliability",
                )
                rel_pipeline = _ReliabilityPipeline(
                    rep_model, models["reliability_head"]
                )
                rel_optimizer = torch.optim.Adam(
                    rel_pipeline.parameters(),
                    lr=(train_cfg.get("optimizer_params", {}) or {}).get("lr", 1e-3),
                )
                rel_out_dim = builder.reliability_out_dim
                rel_loss = (
                    nn.BCEWithLogitsLoss()
                    if rel_out_dim == 1
                    else nn.CrossEntropyLoss()
                )

                rel_checkpoint_path = None
                if kwargs.get("from_last_checkpoint", False):
                    rel_checkpoint_path = _find_latest_checkpoint(reliability_dir)
                rel_start_epoch = _load_checkpoint_if_requested(
                    rel_pipeline, rel_optimizer, rel_checkpoint_path, device
                )

                rel_callbacks = _build_callbacks(train_cfg)
                rel_epochs = int(train_cfg.get("reliability_epochs", 1))
                rel_train_steps = _resolve_steps(
                    train_cfg.get("reliability_train_steps")
                )
                rel_validation_steps = _resolve_steps(
                    train_cfg.get("reliability_validation_steps")
                )

                rel_trainer = Trainer(
                    model=rel_pipeline,
                    train_loader=reliability_loaders["train"],
                    val_loader=reliability_loaders["validation"],
                    loss_fn=rel_loss,
                    optimizer=rel_optimizer,
                    epochs=rel_epochs,
                    device=device,
                    metrics=builder.get_metrics(branch="reliability"),
                    callbacks=rel_callbacks,
                    history_path=str(reliability_dir / "history.json"),
                    branch="reliability",
                    progress_bar=kwargs.get("progress_bar", False),
                    profile=kwargs.get("profile", False),
                    train_steps=rel_train_steps,
                    validation_steps=rel_validation_steps,
                    start_epoch=rel_start_epoch,
                )
                rel_trainer.fit()

        # ================= saving ======================
        if kwargs.get("save_model", False) or kwargs.get("only_save", False):
            save_dir = model_save_path if model_save_path else classifier_dir
            _save_model_checkpoint(
                models["jaeger_model"],
                save_dir,
                "classifier.pt",
                metadata=kwargs.get("meta"),
                num_params=model_num_params,
            )
            if models.get("reliability_head") is not None:
                _save_model_checkpoint(
                    _ReliabilityPipeline(rep_model, models["reliability_head"]),
                    save_dir,
                    "reliability.pt",
                    metadata=kwargs.get("meta"),
                    num_params=model_num_params,
                )

        logger.info("training completed!")
    finally:
        cleanup_distributed()
