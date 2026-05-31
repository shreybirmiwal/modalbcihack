"""Live inference helpers for the autoresearch EEG action model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


CHANNEL_NAMES = ["AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "auto research" / "runs" / "final_model.json"
AUTO_RESEARCH_DIR = Path(__file__).resolve().parents[2] / "auto research"


@dataclass(frozen=True)
class ActionEvent:
    action: str
    label: int
    sample_index: int
    source: str = "bci"

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "label": self.label,
            "sample_index": self.sample_index,
            "source": self.source,
        }


class BCIActionInference:
    """Run the exported autoresearch model on rolling live EEG windows."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        window_samples: int = 32,
        normalization_samples: int = 1250,
        stride_samples: int = 16,
        cooldown_samples: int = 500,
        warmup_samples: int = 1250,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"model artifact not found: {self.model_path}")

        _ensure_auto_research_imports()
        from pipeline import model_from_payload  # type: ignore

        artifact = json.loads(self.model_path.read_text(encoding="utf-8"))
        self.actions = list(artifact["actions"])
        self.model = model_from_payload(artifact["model"])
        self.window_samples = window_samples
        self.normalization_samples = normalization_samples
        self.stride_samples = stride_samples
        self.cooldown_samples = cooldown_samples
        self.warmup_samples = warmup_samples
        self.buffer = np.empty((0, len(CHANNEL_NAMES)), dtype=np.float64)
        self.sample_index = 0
        self.last_prediction_sample = 0
        self.last_emit_sample = -cooldown_samples
        self.last_action = "nothing"

    def update(self, samples: np.ndarray) -> list[ActionEvent]:
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(CHANNEL_NAMES):
            raise ValueError(f"expected samples shaped (n, {len(CHANNEL_NAMES)}), got {arr.shape}")

        self.buffer = np.vstack([self.buffer, arr])
        max_buffer = max(self.normalization_samples, self.window_samples)
        if len(self.buffer) > max_buffer:
            self.buffer = self.buffer[-max_buffer:]
        self.sample_index += len(arr)

        if len(self.buffer) < self.window_samples or self.sample_index < self.warmup_samples:
            return []
        if self.sample_index - self.last_prediction_sample < self.stride_samples:
            return []

        self.last_prediction_sample = self.sample_index
        label = self._predict_latest_label()
        action = self.actions[label]
        self.last_action = action
        if action == "nothing":
            return []
        if self.sample_index - self.last_emit_sample < self.cooldown_samples:
            return []

        self.last_emit_sample = self.sample_index
        return [ActionEvent(action=action, label=label, sample_index=self.sample_index)]

    def _predict_latest_label(self) -> int:
        _ensure_auto_research_imports()
        from pipeline import predict  # type: ignore
        from prepare import Example  # type: ignore

        normalized = self._normalized_window()
        window = normalized.T.tolist()
        example = Example(session=0, frame=self.sample_index, window=window, label=0)
        return int(predict(self.model, [example])[0])

    def _normalized_window(self) -> np.ndarray:
        normalization_buffer = self.buffer[-self.normalization_samples :]
        mean = normalization_buffer.mean(axis=0)
        std = normalization_buffer.std(axis=0)
        std[std == 0.0] = 1.0
        return (self.buffer[-self.window_samples :] - mean) / std


def _ensure_auto_research_imports() -> None:
    path = str(AUTO_RESEARCH_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
