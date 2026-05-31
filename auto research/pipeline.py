"""Agent-editable EEG -> action pipeline.

The autoresearch loop mutates configuration values that affect this pipeline:
window crop, features, artifact clipping, model family, and temporal smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    "action_channels": {
        "1": [1],  # left_squeeze -> AF7
        "2": [0],  # right_squeeze -> AF8
        "3": [2],  # eye_blink -> CHEEK_R
    },
}


@dataclass
class Model:
    config: dict[str, Any]
    means: list[float]
    scales: list[float]
    kind: str
    weights: list[list[float]]
    centroids: list[list[float]]
    detectors: list[dict[str, Any]] = field(default_factory=list)


def train(examples: list[Any], action_count: int, config: dict[str, Any] | None = None) -> Model:
    merged = _merged_config(config)
    if _action_channels(merged) and action_count > 1:
        return _train_channel_ovr(examples, action_count, merged)

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
    if model.kind == "channel_ovr":
        return _predict_channel_ovr(model, examples, config)

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
        "detectors": model.detectors,
    }


def model_from_payload(payload: dict[str, Any]) -> Model:
    return Model(
        config=dict(payload["config"]),
        means=[float(value) for value in payload["means"]],
        scales=[float(value) for value in payload["scales"]],
        kind=str(payload["kind"]),
        weights=[[float(value) for value in row] for row in payload.get("weights", [])],
        centroids=[[float(value) for value in row] for row in payload.get("centroids", [])],
        detectors=list(payload.get("detectors", [])),
    )


def _train_channel_ovr(examples: list[Any], action_count: int, config: dict[str, Any]) -> Model:
    detectors: list[dict[str, Any]] = []
    action_channels = _action_channels(config)
    for label in range(1, action_count):
        channels = action_channels.get(label)
        if not channels:
            continue
        detector_config = dict(config)
        detector_config["channels"] = channels
        rows = [_features(example.window, detector_config) for example in examples]
        labels = [1 if example.label == label else 0 for example in examples]
        standardized, means, scales = _standardize_fit(rows)
        positive_centroid = _binary_centroid(standardized, labels, 1)
        negative_centroid = _binary_centroid(standardized, labels, 0)
        scores = [_centroid_margin(row, positive_centroid, negative_centroid) for row in standardized]
        detectors.append(
            {
                "label": label,
                "channels": channels,
                "config": detector_config,
                "means": means,
                "scales": scales,
                "positive_centroid": positive_centroid,
                "negative_centroid": negative_centroid,
                "threshold": _conservative_binary_threshold(labels, scores),
            }
        )
    return Model(config=config, means=[], scales=[], kind="channel_ovr", weights=[], centroids=[], detectors=detectors)


def _predict_channel_ovr(model: Model, examples: list[Any], config: dict[str, Any] | None = None) -> list[int]:
    predictions = []
    for example in examples:
        best_label = 0
        best_margin = 0.0
        for detector in model.detectors:
            detector_config = dict(detector["config"])
            if config:
                detector_config.update({key: value for key, value in config.items() if key != "channels"})
            row = _features(example.window, detector_config)
            standardized = _standardize_apply(row, detector["means"], detector["scales"])
            score = _centroid_margin(standardized, detector["positive_centroid"], detector["negative_centroid"])
            margin = score - float(detector.get("threshold", 0.0))
            if margin > best_margin:
                best_margin = margin
                best_label = int(detector["label"])
        predictions.append(best_label)
    smooth = int(model.config.get("smooth", 1))
    return _smooth_predictions(predictions, smooth)


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


def _positive_probability(weights: list[list[float]], row: list[float]) -> float:
    probabilities = _softmax(_linear_scores(weights, row))
    return probabilities[1] if len(probabilities) > 1 else 0.0


def _best_binary_threshold(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 1.01
    candidates = sorted(set(scores))
    if not candidates:
        return 0.0
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in candidates:
        predictions = [1 if score >= threshold else 0 for score in scores]
        tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
        fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
        fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def _conservative_binary_threshold(labels: list[int], scores: list[float]) -> float:
    f1_threshold = _best_binary_threshold(labels, scores)
    negative_scores = sorted(score for label, score in zip(labels, scores) if label == 0)
    if not negative_scores:
        return f1_threshold
    index = min(len(negative_scores) - 1, int(0.995 * len(negative_scores)))
    return max(f1_threshold, negative_scores[index])


def _binary_centroid(rows: list[list[float]], labels: list[int], target: int) -> list[float]:
    selected = [row for row, label in zip(rows, labels) if label == target]
    if not selected:
        return [0.0] * (len(rows[0]) if rows else 0)
    width = len(selected[0])
    return [sum(row[idx] for row in selected) / len(selected) for idx in range(width)]


def _centroid_margin(row: list[float], positive_centroid: list[float], negative_centroid: list[float]) -> float:
    positive_distance = sum((value - positive_centroid[idx]) ** 2 for idx, value in enumerate(row))
    negative_distance = sum((value - negative_centroid[idx]) ** 2 for idx, value in enumerate(row))
    return negative_distance - positive_distance


def _action_channels(config: dict[str, Any]) -> dict[int, list[int]]:
    raw = config.get("action_channels", {})
    if not isinstance(raw, dict):
        return {}
    channels: dict[int, list[int]] = {}
    for label, values in raw.items():
        try:
            label_id = int(label)
            channel_values = [int(value) for value in values]
        except (TypeError, ValueError):
            continue
        channels[label_id] = channel_values
    return channels


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
