from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from jaeger.nnlib.pytorch.builder import ModelBuilder


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
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)

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
