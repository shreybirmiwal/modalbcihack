"""Live inference — dead simple raw amplitude spike detection.

If any sample on a channel exceeds 500, that action fires. No normalization,
no z-scores, no ML. Just raw signal > threshold = action.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CHANNEL_NAMES = ["AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "auto research" / "runs" / "final_model.json"

# Raw amplitude threshold — if any sample on the channel exceeds this, fire.
SPIKE_DETECTORS = [
    {"channel": 1, "action": "left_squeeze", "label": 1, "threshold": 700},
    {"channel": 0, "action": "right_squeeze", "label": 2, "threshold": 700},
    {"channel": 2, "action": "eye_blink",     "label": 3, "threshold": 700},
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
    """Raw amplitude spike detector. Signal > 500 = action fires."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        window_samples: int = 32,
        normalization_samples: int = 1250,
        stride_samples: int = 16,
        cooldown_samples: int = 250,
        warmup_samples: int = 0,
        confirm_windows: int = 1,
        blink_zscore_min_peak: float = 2.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.actions = ["nothing", "left_squeeze", "right_squeeze", "eye_blink"]
        self.buffer = np.empty((0, len(CHANNEL_NAMES)), dtype=np.float64)
        self.sample_index = 0
        self.startup_ignore_samples = 2500  # ignore first ~10 seconds at 250Hz
        self.model = True
        self.model_mtime_ns = 0
        # keep these so game controller doesn't crash
        self.window_samples = window_samples
        self.normalization_samples = normalization_samples
        self.stride_samples = stride_samples
        self.cooldown_samples = cooldown_samples
        self.warmup_samples = warmup_samples
        self.confirm_windows = 1
        self.blink_zscore_min_peak = float(blink_zscore_min_peak)
        self.last_action = "nothing"
        self.last_predicted_action = "nothing"
        self.prediction_streak = 0
        self.last_prediction_sample = 0
        self.last_emit_sample = -cooldown_samples
        print(f"[SPIKE DETECTOR] raw amplitude thresholds: {[(d['action'], d['threshold']) for d in SPIKE_DETECTORS]}", flush=True)

    def update(self, samples: np.ndarray) -> list[ActionEvent]:
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(CHANNEL_NAMES):
            raise ValueError(f"expected samples shaped (n, {len(CHANNEL_NAMES)}), got {arr.shape}")

        self.sample_index += len(arr)
        # Ignore first 10 seconds of incoming EEG samples.
        if self.sample_index < self.startup_ignore_samples:
            return []

        # Respect configured cooldown in samples.
        if self.sample_index - self.last_emit_sample < self.cooldown_samples:
            return []

        # Select strongest above-threshold detector in this chunk.
        best_detector: dict[str, Any] | None = None
        best_peak = 0.0
        for det in SPIKE_DETECTORS:
            ch = det["channel"]
            peak = float(np.max(np.abs(arr[:, ch])))
            if peak > det["threshold"] and peak > best_peak:
                best_peak = peak
                best_detector = det

        if best_detector is not None:
            action = str(best_detector["action"])
            label = int(best_detector["label"])
            ch = int(best_detector["channel"])
            self.last_emit_sample = self.sample_index
            print(f"[SPIKE] {action} FIRED! ch={ch} peak={best_peak:.1f} > {best_detector['threshold']}", flush=True)
            return [ActionEvent(action=action, label=label, sample_index=self.sample_index)]

        # Debug: print max values every ~2 seconds
        if self.sample_index % 500 < len(arr):
            peaks = {d["action"]: float(np.max(np.abs(arr[:, d["channel"]]))) for d in SPIKE_DETECTORS}
            print(f"[DEBUG] peaks: {peaks}", flush=True)

        return []

    def reload_model_if_changed(self) -> None:
        pass

    def reload_model(self) -> None:
        pass
