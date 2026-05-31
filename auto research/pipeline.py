"""Agent-editable EEG -> action pipeline.

The autoresearch loop mutates configuration values that affect this pipeline:
window crop, features, artifact clipping, model family, and temporal smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import random


CONFIG: dict[str, Any] = {
    "channels": [0, 1, 2, 3, 4, 5, 6, 7],
    "window_start": 0,
    "window_size": 32,
    "features": ["mean", "std", "energy", "mu", "beta"],
    "clip": 3.0,
    "model": "centroid",
    "epochs": 18,
    "learning_rate": 0.065,
    "l2": 0.0005,
    "smooth": 1,
    "class_weight": "sqrt_balanced",
}


@dataclass
class Model:
    config: dict[str, Any]
    means: list[float]
    scales: list[float]
    kind: str
    weights: list[list[float]]
    centroids: list[list[float]]


def train(examples: list[Any], action_count: int, config: dict[str, Any] | None = None) -> Model:
    merged = _merged_config(config)
    rows = [_features(example.window, merged) for example in examples]
    labels = [example.label for example in examples]
    standardized, means, scales = _standardize_fit(rows)
    kind = str(merged.get("model", "centroid"))

    if kind == "softmax":
        weights = _train_softmax(standardized, labels, action_count, merged)
        centroids: list[list[float]] = []
    elif kind == "perceptron":
        weights = _train_perceptron(standardized, labels, action_count, merged)
        centroids = []
    else:
        weights = []
        centroids = _train_centroids(standardized, labels, action_count)
        kind = "centroid"

    return Model(config=merged, means=means, scales=scales, kind=kind, weights=weights, centroids=centroids)


def predict(model: Model, examples: list[Any], config: dict[str, Any] | None = None) -> list[int]:
    merged = dict(model.config)
    if config:
        merged.update(config)
    rows = [_standardize_apply(_features(example.window, merged), model.means, model.scales) for example in examples]
    predictions = [_predict_one(model, row) for row in rows]
    smooth = int(merged.get("smooth", 1))
    return _smooth_predictions(predictions, smooth)


def model_to_payload(model: Model) -> dict[str, Any]:
    return {
        "config": model.config,
        "means": model.means,
        "scales": model.scales,
        "kind": model.kind,
        "weights": model.weights,
        "centroids": model.centroids,
    }


def model_from_payload(payload: dict[str, Any]) -> Model:
    return Model(
        config=dict(payload["config"]),
        means=[float(value) for value in payload["means"]],
        scales=[float(value) for value in payload["scales"]],
        kind=str(payload["kind"]),
        weights=[[float(value) for value in row] for row in payload.get("weights", [])],
        centroids=[[float(value) for value in row] for row in payload.get("centroids", [])],
    )


def _features(window: list[list[float]], config: dict[str, Any]) -> list[float]:
    selected_channels = list(config.get("channels", CONFIG["channels"]))
    start = int(config.get("window_start", 0))
    size = int(config.get("window_size", 32))
    feature_names = set(config.get("features", CONFIG["features"]))
    clip = float(config.get("clip", 0.0) or 0.0)
    vector: list[float] = []

    for channel in selected_channels:
        samples = window[channel][start : start + size]
        if not samples:
            samples = window[channel]
        if clip > 0:
            samples = [max(-clip, min(clip, value)) for value in samples]
        mean = sum(samples) / len(samples)
        centered = [value - mean for value in samples]
        variance = sum(value * value for value in centered) / len(samples)

        if "mean" in feature_names:
            vector.append(mean)
        if "std" in feature_names:
            vector.append(math.sqrt(variance + 1e-9))
        if "energy" in feature_names:
            vector.append(sum(value * value for value in samples) / len(samples))
        if "slope" in feature_names:
            vector.append((samples[-1] - samples[0]) / max(1, len(samples) - 1))
        if "mu" in feature_names:
            vector.append(_band_energy(samples, 10.0))
            vector.append(_band_energy(samples, 12.0))
        if "beta" in feature_names:
            vector.append(_band_energy(samples, 20.0))
            vector.append(_band_energy(samples, 26.0))

    if "global_energy" in feature_names:
        all_samples = [value for channel in selected_channels for value in window[channel][start : start + size]]
        vector.append(sum(value * value for value in all_samples) / max(1, len(all_samples)))

    return vector


def _band_energy(samples: list[float], frequency: float) -> float:
    sin_total = 0.0
    cos_total = 0.0
    sample_rate = 128.0
    for idx, value in enumerate(samples):
        angle = 2.0 * math.pi * frequency * idx / sample_rate
        sin_total += value * math.sin(angle)
        cos_total += value * math.cos(angle)
    return (sin_total * sin_total + cos_total * cos_total) / max(1, len(samples) * len(samples))


def _standardize_fit(rows: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    width = len(rows[0]) if rows else 0
    means = [sum(row[idx] for row in rows) / len(rows) for idx in range(width)]
    scales = []
    for idx, mean in enumerate(means):
        variance = sum((row[idx] - mean) ** 2 for row in rows) / len(rows)
        scales.append(math.sqrt(variance) or 1.0)
    return [[(value - means[idx]) / scales[idx] for idx, value in enumerate(row)] for row in rows], means, scales


def _standardize_apply(row: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - means[idx]) / scales[idx] for idx, value in enumerate(row)]


def _train_centroids(rows: list[list[float]], labels: list[int], action_count: int) -> list[list[float]]:
    width = len(rows[0]) if rows else 0
    centroids = [[0.0] * width for _ in range(action_count)]
    counts = [0] * action_count
    for row, label in zip(rows, labels):
        counts[label] += 1
        for idx, value in enumerate(row):
            centroids[label][idx] += value
    for label in range(action_count):
        denom = max(1, counts[label])
        centroids[label] = [value / denom for value in centroids[label]]
    return centroids


def _train_perceptron(rows: list[list[float]], labels: list[int], action_count: int, config: dict[str, Any]) -> list[list[float]]:
    width = len(rows[0]) if rows else 0
    weights = [[0.0] * (width + 1) for _ in range(action_count)]
    lr = float(config.get("learning_rate", 0.05))
    epochs = int(config.get("epochs", 12))
    label_weights = _class_weights(labels, action_count, config)
    order = list(range(len(rows)))
    rng = random.Random(7)
    for _ in range(epochs):
        rng.shuffle(order)
        for row_idx in order:
            row = rows[row_idx]
            label = labels[row_idx]
            sample_weight = label_weights[label]
            pred = _argmax(_linear_scores(weights, row))
            if pred != label:
                for idx, value in enumerate(row):
                    weights[label][idx] += lr * sample_weight * value
                    weights[pred][idx] -= lr * sample_weight * value
                weights[label][-1] += lr * sample_weight
                weights[pred][-1] -= lr * sample_weight
    return weights


def _train_softmax(rows: list[list[float]], labels: list[int], action_count: int, config: dict[str, Any]) -> list[list[float]]:
    width = len(rows[0]) if rows else 0
    weights = [[0.0] * (width + 1) for _ in range(action_count)]
    lr = float(config.get("learning_rate", 0.05))
    l2 = float(config.get("l2", 0.0))
    epochs = int(config.get("epochs", 20))
    label_weights = _class_weights(labels, action_count, config)
    order = list(range(len(rows)))
    rng = random.Random(11)

    for _ in range(epochs):
        rng.shuffle(order)
        for row_idx in order:
            row = rows[row_idx]
            label = labels[row_idx]
            sample_weight = label_weights[label]
            scores = _linear_scores(weights, row)
            probs = _softmax(scores)
            for action, prob in enumerate(probs):
                gradient = sample_weight * (prob - (1.0 if action == label else 0.0))
                for idx, value in enumerate(row):
                    weights[action][idx] -= lr * (gradient * value + l2 * weights[action][idx])
                weights[action][-1] -= lr * gradient
    return weights


def _class_weights(labels: list[int], action_count: int, config: dict[str, Any]) -> list[float]:
    mode = str(config.get("class_weight", "none"))
    if mode in {"", "none", "0", "false"}:
        return [1.0] * action_count
    counts = [0] * action_count
    for label in labels:
        counts[label] += 1
    total = max(1, len(labels))
    exponent = 0.5 if mode == "sqrt_balanced" else 1.0
    weights = []
    for count in counts:
        if count == 0:
            weights.append(1.0)
            continue
        weights.append((total / (action_count * count)) ** exponent)
    return weights


def _predict_one(model: Model, row: list[float]) -> int:
    if model.kind == "centroid":
        distances = []
        for centroid in model.centroids:
            distances.append(sum((value - centroid[idx]) ** 2 for idx, value in enumerate(row)))
        return min(range(len(distances)), key=lambda idx: distances[idx])
    return _argmax(_linear_scores(model.weights, row))


def _linear_scores(weights: list[list[float]], row: list[float]) -> list[float]:
    return [sum(weight[idx] * value for idx, value in enumerate(row)) + weight[-1] for weight in weights]


def _softmax(scores: list[float]) -> list[float]:
    offset = max(scores)
    exps = [math.exp(score - offset) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _smooth_predictions(predictions: list[int], width: int) -> list[int]:
    if width <= 1:
        return predictions
    smoothed: list[int] = []
    for idx in range(len(predictions)):
        start = max(0, idx - width + 1)
        recent = predictions[start : idx + 1]
        counts: dict[int, int] = {}
        for pred in recent:
            counts[pred] = counts.get(pred, 0) + 1
        smoothed.append(max(counts, key=lambda label: (counts[label], -label)))
    return smoothed


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(CONFIG)
    if config:
        merged.update(config)
    return merged
