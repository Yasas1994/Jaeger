from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from jaeger.nnlib.pytorch.builder import ModelBuilder


def _dummy_input_from_config(
    config: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a dummy (x, mask) tuple that matches the model input shape."""
    model_cfg = config.get("model", {})
    embedding_cfg = model_cfg.get("embedding", {})
    sp_cfg = model_cfg.get("string_processor", {})
    frames = int(embedding_cfg.get("frames", 6))
    crop_size = int(sp_cfg.get("crop_size", 500))
    x = torch.zeros((1, frames, crop_size), dtype=torch.long)
    mask = torch.ones((1, frames, crop_size), dtype=torch.bool)
    return x, mask


class PyTorchInferenceRunner:
    """Load and run a PyTorch Jaeger model for inference."""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        builder = ModelBuilder(config)
        self.models = builder.build_fragment_classifier()
        self.model = self.models["jaeger_model"]
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, path: Union[str, Path]):
        path = Path(path)
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=self.device, weights_only=False)
        sd = state.get("model_state_dict", state)
        try:
            self.model.load_state_dict(sd)
        except RuntimeError:
            # The checkpoint may contain trained MaskedConv1D weights that
            # require lazy layers to be materialized before loading.
            self._initialize_lazy_layers()
            self.model.load_state_dict(sd)

    def _initialize_lazy_layers(self):
        """Run a dummy forward so MaskedConv1D layers materialize before loading."""
        dummy_x, dummy_mask = _dummy_input_from_config(self.config)
        dummy_x = dummy_x.to(self.device)
        dummy_mask = dummy_mask.to(self.device)
        with torch.no_grad():
            self.model(dummy_x, dummy_mask)

    def predict(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        batch_size: int = 32,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch_x = x[i : i + batch_size].to(self.device)
                batch_mask = (
                    None if mask is None else mask[i : i + batch_size].to(self.device)
                )
                out = self.model(batch_x, mask=batch_mask)
                outputs.append({k: v.cpu() for k, v in out.items()})

        return {k: torch.cat([o[k] for o in outputs], dim=0) for k in outputs[0]}

    def predict_from_dataset(
        self, dataset: torch.utils.data.Dataset, batch_size: int = 32
    ) -> Dict[str, torch.Tensor]:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        outputs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                mask = batch[2].to(self.device) if len(batch) > 2 else None
                out = self.model(x, mask=mask)
                outputs.append({k: v.cpu() for k, v in out.items()})
        return {k: torch.cat([o[k] for o in outputs], dim=0) for k in outputs[0]}
