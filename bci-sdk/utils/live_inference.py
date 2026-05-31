"""Live inference — simple spike detection on z-scored EEG channels.

No model files needed. Each action has one channel. If the signal on that
channel spikes above the baseline (measured via a rolling z-score), fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CHANNEL_NAMES = ["AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "auto research" / "runs" / "final_model.json"

# Channel index -> action name, label
# AF7 (idx 1) = left_squeeze, AF8 (idx 0) = right_squeeze, CHEEK_R (idx 2) = eye_blink
SPIKE_DETECTORS = [
    {"channel": 1, "action": "left_squeeze",  "label": 1, "threshold": 1.35},
    {"channel": 0, "action": "right_squeeze", "label": 2, "threshold": 1.35},
    {"channel": 2, "action": "eye_blink",     "label": 3, "threshold": 1.35},
]


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
    """Simple spike detector on rolling z-scored EEG windows."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        window_samples: int = 32,
        normalization_samples: int = 1250,
        stride_samples: int = 16,
        cooldown_samples: int = 500,
        warmup_samples: int = 1250,
        confirm_windows: int = 2,
        blink_zscore_min_peak: float = 2.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.actions = ["nothing", "left_squeeze", "right_squeeze", "eye_blink"]
        self.window_samples = window_samples
        self.normalization_samples = normalization_samples
        self.stride_samples = stride_samples
        self.cooldown_samples = cooldown_samples
        self.warmup_samples = warmup_samples
        self.confirm_windows = max(1, int(confirm_windows))
        self.blink_zscore_min_peak = float(blink_zscore_min_peak)
        self.buffer = np.empty((0, len(CHANNEL_NAMES)), dtype=np.float64)
        self.sample_index = 0
        self.last_prediction_sample = 0
        self.last_emit_sample = -cooldown_samples
        self.last_action = "nothing"
        self.last_predicted_action = "nothing"
        self.prediction_streak = 0
        self.awaiting_nothing_reset = False
        self.model = True  # dummy so game controller doesn't complain
        self.model_mtime_ns = 0
        print(f"[SPIKE DETECTOR] thresholds: {[(d['action'], d['threshold']) for d in SPIKE_DETECTORS]}", flush=True)

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
        label = self._detect_spike()
        action = self.actions[label]
        self.last_action = action

        if action == "nothing":
            self.prediction_streak = 0
            self.awaiting_nothing_reset = False
            self.last_predicted_action = action
            return []

        if action == self.last_predicted_action:
            self.prediction_streak += 1
        else:
            self.prediction_streak = 1
        self.last_predicted_action = action
        if self.prediction_streak < self.confirm_windows:
            return []

        if self.awaiting_nothing_reset:
            return []
        if self.sample_index - self.last_emit_sample < self.cooldown_samples:
            return []

        self.last_emit_sample = self.sample_index
        self.awaiting_nothing_reset = True
        return [ActionEvent(action=action, label=label, sample_index=self.sample_index)]

    def _detect_spike(self) -> int:
        """Check each channel — if z-scored energy spikes above threshold, fire."""
        normalized = self._normalized_window()
        best_label = 0
        best_strength = 0.0

        for det in SPIKE_DETECTORS:
            ch = det["channel"]
            channel_data = normalized[:, ch]
            # spike score: mean of absolute values (how active is this channel?)
            mean_abs = float(np.mean(np.abs(channel_data)))
            peak = float(np.max(np.abs(channel_data)))
            score = mean_abs + 0.5 * peak

            if score > det["threshold"] and score > best_strength:
                best_strength = score
                best_label = det["label"]

        return best_label

    def _normalized_window(self) -> np.ndarray:
        """Z-score the latest window against the rolling baseline."""
        baseline = self.buffer[-self.normalization_samples:]
        mean = baseline.mean(axis=0)
        std = baseline.std(axis=0)
        std[std == 0.0] = 1.0
        window = self.buffer[-self.window_samples:]
        return (window - mean) / std

    def reload_model_if_changed(self) -> None:
        pass

    def reload_model(self) -> None:
        pass
