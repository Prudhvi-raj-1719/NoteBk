"""OSNet x1.0 embeddings via torchreid.reid (Market-1501 weights)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from configs.reid_config import (
    OSNET_X1_0_MARKET1501_URL,
    REID_DEVICE,
    REID_MODEL_NAME,
)

_MARKET1501_WEIGHT_FILE = "osnet_x1_0_market1501.pt"


class OsnetEmbedder:
    """Lazy-loaded OSNet feature extractor (torchreid.reid.models.osnet)."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or REID_DEVICE
        self.model_name = REID_MODEL_NAME
        self._model = None
        self._transform = None
        self._ready = False

    def _download_market1501_weights(self, cache_path: Path) -> None:
        import gdown

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            print(f"[REID] Downloading Market-1501 weights -> {cache_path}")
            gdown.download(OSNET_X1_0_MARKET1501_URL, str(cache_path), quiet=False)

    def _ensure_model(self) -> None:
        if self._ready:
            return

        import torch
        import torch.nn.functional as F
        from torchreid.reid.models.osnet import osnet_x1_0
        from torchvision import transforms

        if self.model_name != "osnet_x1_0":
            raise ValueError(f"Unsupported REID model: {self.model_name} (use osnet_x1_0)")

        cache_dir = Path.home() / ".cache" / "notebk_reid"
        weight_path = cache_dir / _MARKET1501_WEIGHT_FILE

        model = osnet_x1_0(num_classes=751, pretrained=False, loss="softmax")
        self._download_market1501_weights(weight_path)
        state = torch.load(weight_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        model.eval()
        model.to(self.device)

        self._torch = torch
        self._F = F
        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._ready = True
        print(
            f"[REID] OSNet loaded: model={self.model_name} device={self.device} "
            "weights=Market-1501"
        )

    def embed_crops(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Return L2-normalized embeddings shape (N, D). Empty input -> (0, 512)."""
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        self._ensure_model()
        tensors = []
        for crop in crops_bgr:
            if crop is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(rgb))

        if not tensors:
            return np.zeros((0, 512), dtype=np.float32)

        batch = self._torch.stack(tensors).to(self.device)
        with self._torch.no_grad():
            features = self._model(batch)
        features = self._F.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype(np.float32)
