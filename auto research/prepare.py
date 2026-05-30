"""Frozen data preparation and evaluation harness.

This file is intentionally treated as immutable by the autoresearch loop. It
creates deterministic synthetic EEG/game logs, builds session-based splits, and
scores candidate pipelines with nested CV plus a held-out overfit guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import hashlib
import importlib
import math
import random


ACTIONS = ["noop", "jump", "left", "right"]
SAMPLE_RATE_HZ = 128
SAMPLES_PER_WINDOW = 32
CHANNELS = 8
FRAMES_PER_SESSION = 150
SESSIONS = 6


@dataclass(frozen=True)
class Example:
    session: int
    frame: int
    window: list[list[float]]
    label: int


@dataclass(frozen=True)
class Dataset:
    subject: str
    stage: int
    actions: list[str]
    examples: list[Example]
    train_sessions: list[int]
    heldout_session: int
    sealed_session: int


@dataclass(frozen=True)
class Metrics:
    closed_loop_score: float
    balanced_accuracy: float
    macro_f1: float
    latency_cost: float

    @property
    def metric(self) -> float:
        return self.closed_loop_score + 0.20 * self.balanced_accuracy + 0.10 * self.macro_f1


@dataclass(frozen=True)
class Evaluation:
    cv: Metrics
    heldout: Metrics
    sealed: Metrics | None
    generalization_gap: float
    reward_base: float


def action_set(stage: int) -> list[str]:
    if stage < 2 or stage > len(ACTIONS):
        raise ValueError(f"stage must be between 2 and {len(ACTIONS)}")
    return ACTIONS[:stage]


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_dataset(subject: str = "S03", stage: int = 2) -> Dataset:
    actions = action_set(stage)
    rng = random.Random(stable_seed("dataset", subject, stage))
    subject_profile = _subject_profile(subject, len(actions))
    examples: list[Example] = []

    for session in range(SESSIONS):
        labels = _session_labels(rng, len(actions), FRAMES_PER_SESSION)
        drift = rng.uniform(-0.18, 0.18)
        noise_scale = 0.62 + session * 0.035
        for frame, label in enumerate(labels):
            prev_label = labels[frame - 1] if frame else 0
            window = _window_for_label(
                rng=rng,
                label=label,
                prev_label=prev_label,
                profile=subject_profile,
                drift=drift,
                noise_scale=noise_scale,
            )
            examples.append(Example(session=session, frame=frame, window=window, label=label))

    return Dataset(
        subject=subject,
        stage=stage,
        actions=actions,
        examples=examples,
        train_sessions=[0, 1, 2, 3],
        heldout_session=4,
        sealed_session=5,
    )


def evaluate_config(
    config: dict[str, Any],
    subject: str = "S03",
    stage: int = 2,
    sealed: bool = False,
    pipeline_module: str = "pipeline",
) -> Evaluation:
    dataset = load_dataset(subject=subject, stage=stage)
    pipeline = importlib.import_module(pipeline_module)

    fold_metrics: list[Metrics] = []
    for validation_session in dataset.train_sessions:
        train_sessions = [s for s in dataset.train_sessions if s != validation_session]
        train_examples = _examples_for_sessions(dataset, train_sessions)
        val_examples = _examples_for_sessions(dataset, [validation_session])
        model = pipeline.train(train_examples, len(dataset.actions), config)
        predictions = pipeline.predict(model, val_examples, config)
        fold_metrics.append(score_predictions(val_examples, predictions, config))

    cv = _average_metrics(fold_metrics)

    train_examples = _examples_for_sessions(dataset, dataset.train_sessions)
    heldout_examples = _examples_for_sessions(dataset, [dataset.heldout_session])
    model = pipeline.train(train_examples, len(dataset.actions), config)
    heldout_predictions = pipeline.predict(model, heldout_examples, config)
    heldout = score_predictions(heldout_examples, heldout_predictions, config)

    sealed_metrics = None
    if sealed:
        sealed_examples = _examples_for_sessions(dataset, [dataset.sealed_session])
        sealed_predictions = pipeline.predict(model, sealed_examples, config)
        sealed_metrics = score_predictions(sealed_examples, sealed_predictions, config)

    gap = max(0.0, cv.metric - heldout.metric)
    reward_base = heldout.closed_loop_score + 0.20 * heldout.balanced_accuracy + 0.10 * heldout.macro_f1
    return Evaluation(cv=cv, heldout=heldout, sealed=sealed_metrics, generalization_gap=gap, reward_base=reward_base)


def score_predictions(examples: list[Example], predictions: list[int], config: dict[str, Any]) -> Metrics:
    if len(examples) != len(predictions):
        raise ValueError("prediction count must match example count")
    labels = [example.label for example in examples]
    action_count = max(labels + predictions) + 1 if labels else 1
    balanced = _balanced_accuracy(labels, predictions, action_count)
    macro_f1 = _macro_f1(labels, predictions, action_count)
    latency_cost = 0.002 * float(config.get("window_start", 0)) + 0.001 * float(config.get("smooth", 0))
    closed_loop = _closed_loop_score(labels, predictions, action_count) - latency_cost
    return Metrics(
        closed_loop_score=max(0.0, closed_loop),
        balanced_accuracy=balanced,
        macro_f1=macro_f1,
        latency_cost=latency_cost,
    )


def _examples_for_sessions(dataset: Dataset, sessions: Iterable[int]) -> list[Example]:
    wanted = set(sessions)
    return [example for example in dataset.examples if example.session in wanted]


def _subject_profile(subject: str, action_count: int) -> list[list[float]]:
    rng = random.Random(stable_seed("profile", subject))
    profile: list[list[float]] = []
    for action in range(action_count):
        row = []
        for channel in range(CHANNELS):
            base = 0.25 + 0.08 * action + 0.03 * channel
            row.append(base + rng.uniform(-0.20, 0.20))
        profile.append(row)
    return profile


def _session_labels(rng: random.Random, action_count: int, n: int) -> list[int]:
    labels: list[int] = []
    current = 0
    remaining = 0
    weights = [0.56] + [0.44 / (action_count - 1)] * (action_count - 1)
    for _ in range(n):
        if remaining <= 0:
            current = _weighted_choice(rng, weights)
            remaining = rng.randint(3, 11 if current == 0 else 7)
        labels.append(current)
        remaining -= 1
    return labels


def _weighted_choice(rng: random.Random, weights: list[float]) -> int:
    threshold = rng.random() * sum(weights)
    total = 0.0
    for idx, weight in enumerate(weights):
        total += weight
        if total >= threshold:
            return idx
    return len(weights) - 1


def _window_for_label(
    rng: random.Random,
    label: int,
    prev_label: int,
    profile: list[list[float]],
    drift: float,
    noise_scale: float,
) -> list[list[float]]:
    frequencies = [8.0, 12.0, 20.0, 26.0]
    window: list[list[float]] = []
    artifact = rng.random() < 0.035
    for channel in range(CHANNELS):
        samples: list[float] = []
        weight = profile[label][channel]
        prev_weight = 0.35 * profile[prev_label][channel]
        phase = 0.15 * channel + drift
        for sample in range(SAMPLES_PER_WINDOW):
            t = sample / SAMPLE_RATE_HZ
            signal = weight * math.sin(2 * math.pi * frequencies[label] * t + phase)
            signal += prev_weight * math.sin(2 * math.pi * frequencies[prev_label] * t + phase / 2)
            signal += 0.18 * math.sin(2 * math.pi * 2.0 * t + drift)
            signal += rng.gauss(0.0, noise_scale)
            if artifact and channel in (0, 1):
                signal += rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 2.4)
            samples.append(signal)
        window.append(samples)
    return window


def _balanced_accuracy(labels: list[int], predictions: list[int], action_count: int) -> float:
    recalls = []
    for action in range(action_count):
        total = sum(1 for label in labels if label == action)
        if total == 0:
            continue
        correct = sum(1 for label, pred in zip(labels, predictions) if label == action and pred == action)
        recalls.append(correct / total)
    return sum(recalls) / len(recalls) if recalls else 0.0


def _macro_f1(labels: list[int], predictions: list[int], action_count: int) -> float:
    values = []
    for action in range(action_count):
        tp = sum(1 for label, pred in zip(labels, predictions) if label == action and pred == action)
        fp = sum(1 for label, pred in zip(labels, predictions) if label != action and pred == action)
        fn = sum(1 for label, pred in zip(labels, predictions) if label == action and pred != action)
        denom = 2 * tp + fp + fn
        values.append((2 * tp / denom) if denom else 0.0)
    return sum(values) / len(values) if values else 0.0


def _closed_loop_score(labels: list[int], predictions: list[int], action_count: int) -> float:
    score = 0.0
    combo = 0
    energy = 1.0
    for label, pred in zip(labels, predictions):
        if pred == label:
            combo += 1
            score += 1.0 + min(combo, 12) * 0.035
            energy = min(1.25, energy + 0.012)
        else:
            combo = 0
            penalty = 1.25 if label != 0 else 0.70
            score -= penalty
            energy -= 0.030 if pred != 0 else 0.018
        if action_count > 2 and label in (2, 3) and pred == label:
            score += 0.20
        if energy <= 0.0:
            break
    return max(0.0, score / max(1, len(labels)))


def _average_metrics(metrics: list[Metrics]) -> Metrics:
    count = len(metrics)
    return Metrics(
        closed_loop_score=sum(m.closed_loop_score for m in metrics) / count,
        balanced_accuracy=sum(m.balanced_accuracy for m in metrics) / count,
        macro_f1=sum(m.macro_f1 for m in metrics) / count,
        latency_cost=sum(m.latency_cost for m in metrics) / count,
    )
