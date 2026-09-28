from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import random

import numpy as np
import torch
from torchvision.models.video import r3d_18

from .video import LABELS, PREPROCESS, sample_frames, sha256


def tensor_from_frames(frames: np.ndarray, augment=False):
    # Keep temporal ordering. The same spatial crop/flip is applied to every frame.
    size = PREPROCESS["crop"]
    h, w = frames.shape[1:3]
    y = random.randint(0, h-size) if augment else (h-size)//2
    x = random.randint(0, w-size) if augment else (w-size)//2
    frames = frames[:, y:y+size, x:x+size]
    if augment and random.random() < .5:
        frames = frames[:, :, ::-1]
    tensor = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float().div_(255)
    mean = tensor.new_tensor(PREPROCESS["mean"])[:, None, None, None]
    std = tensor.new_tensor(PREPROCESS["std"])[:, None, None, None]
    return (tensor-mean)/std


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


class Predictor:
    def __init__(self, checkpoint: Path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = Path(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["labels"] != LABELS or saved["preprocess"] != PREPROCESS:
            raise ValueError("Checkpoint labels or preprocessing differ from this application")
        self.model = r3d_18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(LABELS))
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()
        self.version = sha256(checkpoint)
        self.thresholds = saved["thresholds"]

    @torch.inference_mode()
    def predict_frames(self, frames):
        x = torch.stack([tensor_from_frames(f) for f in frames]).to(self.device)
        with autocast(self.device):
            logits = self.model(x)
        return logits.float().softmax(1).cpu().numpy()

    def predict_window(self, path, start, end):
        return self.predict_frames([sample_frames(Path(path), start, end-start)])[0]
