"""Autoresearch ratchet loop for the EEG -> game-control demo."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from prepare import evaluate_config
from pipeline import CONFIG as BASE_CONFIG


RUNS_DIR = Path("runs")
LOG_PATH = RUNS_DIR / "research_log.jsonl"
BEST_PATH = RUNS_DIR / "best_pipeline_config.json"
PROGRAM_PATH = Path("program.md")


@dataclass(frozen=True)
class Candidate:
    config: dict[str, Any]
    hypothesis: str
    signature: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an autoresearch sweep over EEG decoding pipelines.")
    parser.add_argument("--subject", default="S03")
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sealed-probe-every", type=int, default=3)
    parser.add_argument("--gap-tolerance", type=float, default=0.18)
    args = parser.parse_args()

    RUNS_DIR.mkdir(exist_ok=True)
    program_text = load_program_text()
    program_hash = hashlib.sha256(program_text.encode("utf-8")).hexdigest()[:12]
    tried = load_tried_signatures()
    best_config, best_reward = load_best(args.subject, args.stage)

    print(f"Autoresearch EEG loop | subject={args.subject} stage={args.stage}")
    print(f"program={PROGRAM_PATH} hash={program_hash} goal={extract_goal(program_text)}")
    print(f"starting_best_reward={best_reward:.4f} tried={len(tried)}")

    for round_idx in range(1, args.rounds + 1):
        candidates = propose_batch(
            base_config=best_config,
            tried=tried,
            batch_size=args.batch_size,
            round_idx=round_idx,
            subject=args.subject,
            stage=args.stage,
        )
        sealed = args.sealed_probe_every > 0 and round_idx % args.sealed_probe_every == 0
        print(f"\nround={round_idx} candidates={len(candidates)} sealed_probe={sealed}")

        results = evaluate_candidates(candidates, args.subject, args.stage, args.workers, sealed, program_hash)
        accepted = False
        for result in sorted(results, key=lambda row: row["reward"], reverse=True):
            tried.add(result["signature"])
            append_log(result)
            if result["reward"] > best_reward and result["generalization_gap"] <= args.gap_tolerance:
                best_reward = result["reward"]
                best_config = result["config"]
                save_best(best_config, best_reward, args.subject, args.stage, result)
                accepted = True
                print(
                    "accepted "
                    f"reward={best_reward:.4f} heldout={result['heldout']['closed_loop_score']:.4f} "
                    f"gap={result['generalization_gap']:.4f} hypothesis={result['hypothesis']}"
                )
                break

        if not accepted:
            top = max(results, key=lambda row: row["reward"])
            print(
                "no_accept "
                f"top_reward={top['reward']:.4f} best={best_reward:.4f} "
                f"gap={top['generalization_gap']:.4f}"
            )

    print(f"\nBest config saved to {BEST_PATH}")
    print(json.dumps({"reward": best_reward, "config": best_config}, indent=2, sort_keys=True))


def propose_batch(
    base_config: dict[str, Any],
    tried: set[str],
    batch_size: int,
    round_idx: int,
    subject: str,
    stage: int,
) -> list[Candidate]:
    rng = random.Random(stable_seed("proposal", subject, stage, round_idx, len(tried)))
    proposals: list[Candidate] = []
    attempts = 0
    while len(proposals) < batch_size and attempts < batch_size * 40:
        attempts += 1
        config = dict(base_config)
        move = rng.choice(
            [
                "feature_family",
                "clip",
                "model",
                "channels",
                "window",
                "smooth",
                "learning",
                "global_energy",
            ]
        )
        hypothesis = mutate(config, move, rng)
        signature = config_signature(config)
        if signature in tried or any(candidate.signature == signature for candidate in proposals):
            continue
        proposals.append(Candidate(config=config, hypothesis=hypothesis, signature=signature))
    return proposals


def mutate(config: dict[str, Any], move: str, rng: random.Random) -> str:
    if move == "feature_family":
        choices = [
            ["mean", "std", "energy", "mu", "beta"],
            ["std", "energy", "mu", "beta", "slope"],
            ["energy", "mu", "beta", "global_energy"],
            ["mean", "std", "mu"],
            ["std", "beta", "slope"],
        ]
        config["features"] = rng.choice(choices)
        return "change feature family to test which motor-band summaries survive session drift"
    if move == "clip":
        config["clip"] = rng.choice([1.8, 2.2, 2.8, 3.4, 4.0, 0.0])
        return "adjust artifact clipping to trade blink robustness against signal loss"
    if move == "model":
        config["model"] = rng.choice(["centroid", "perceptron", "softmax"])
        config["epochs"] = rng.choice([10, 16, 24, 32])
        return "swap classifier family while preserving the frozen evaluation harness"
    if move == "channels":
        channel_sets = [
            [0, 1, 2, 3],
            [2, 3, 4, 5, 6, 7],
            [0, 2, 4, 6],
            [1, 3, 5, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ]
        config["channels"] = rng.choice(channel_sets)
        return "search channel subsets because subject-specific EEG topographies differ"
    if move == "window":
        config["window_start"] = rng.choice([0, 2, 4, 6, 8])
        config["window_size"] = rng.choice([20, 24, 28, 32])
        if config["window_start"] + config["window_size"] > 32:
            config["window_start"] = 32 - config["window_size"]
        return "crop the causal EEG window to test latency versus evidence accumulation"
    if move == "smooth":
        config["smooth"] = rng.choice([1, 2, 3, 4, 5])
        return "vary temporal smoothing to improve game score without hiding fast controls"
    if move == "learning":
        config["learning_rate"] = rng.choice([0.025, 0.04, 0.065, 0.09, 0.12])
        config["l2"] = rng.choice([0.0, 0.0002, 0.0005, 0.001, 0.003])
        config["model"] = rng.choice(["perceptron", "softmax"])
        return "tune optimizer pressure for small noisy EEG folds"
    features = list(config.get("features", []))
    if "global_energy" in features:
        features.remove("global_energy")
    else:
        features.append("global_energy")
    config["features"] = features
    return "toggle global energy as a low-cost artifact and effort feature"


def evaluate_candidates(
    candidates: list[Candidate],
    subject: str,
    stage: int,
    workers: int,
    sealed: bool,
    program_hash: str,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_evaluate_one(candidate, subject, stage, sealed, program_hash) for candidate in candidates]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_evaluate_one, candidate, subject, stage, sealed, program_hash) for candidate in candidates]
        return [future.result() for future in as_completed(futures)]


def _evaluate_one(candidate: Candidate, subject: str, stage: int, sealed: bool, program_hash: str) -> dict[str, Any]:
    started = time.time()
    evaluation = evaluate_config(candidate.config, subject=subject, stage=stage, sealed=sealed)
    experiments_seen = max(1, len(load_tried_signatures()))
    multiple_comparison_penalty = 0.006 * math.log1p(experiments_seen)
    reward = evaluation.reward_base - 0.55 * evaluation.generalization_gap - multiple_comparison_penalty
    return {
        "timestamp": round(time.time(), 3),
        "subject": subject,
        "stage": stage,
        "program_hash": program_hash,
        "signature": candidate.signature,
        "hypothesis": candidate.hypothesis,
        "config": candidate.config,
        "cv": asdict(evaluation.cv),
        "heldout": asdict(evaluation.heldout),
        "sealed": asdict(evaluation.sealed) if evaluation.sealed else None,
        "generalization_gap": evaluation.generalization_gap,
        "reward": reward,
        "seconds": round(time.time() - started, 3),
    }


def load_best(subject: str, stage: int) -> tuple[dict[str, Any], float]:
    if BEST_PATH.exists():
        payload = json.loads(BEST_PATH.read_text())
        if payload.get("subject") == subject and payload.get("stage") == stage:
            return payload["config"], float(payload["reward"])
    baseline = dict(BASE_CONFIG)
    evaluation = evaluate_config(baseline, subject=subject, stage=stage, sealed=True)
    reward = evaluation.reward_base - 0.55 * evaluation.generalization_gap
    save_best(
        baseline,
        reward,
        subject,
        stage,
        {
            "hypothesis": "baseline pipeline",
            "cv": asdict(evaluation.cv),
            "heldout": asdict(evaluation.heldout),
            "sealed": asdict(evaluation.sealed) if evaluation.sealed else None,
            "generalization_gap": evaluation.generalization_gap,
        },
    )
    return baseline, reward


def save_best(config: dict[str, Any], reward: float, subject: str, stage: int, result: dict[str, Any]) -> None:
    payload = {
        "subject": subject,
        "stage": stage,
        "reward": reward,
        "config": config,
        "accepted_from": result,
    }
    BEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_log(result: dict[str, Any]) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")


def load_tried_signatures() -> set[str]:
    if not LOG_PATH.exists():
        return set()
    signatures = set()
    for line in LOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            signatures.add(json.loads(line)["signature"])
        except (KeyError, json.JSONDecodeError):
            continue
    return signatures


def load_program_text() -> str:
    if not PROGRAM_PATH.exists():
        return ""
    return PROGRAM_PATH.read_text(encoding="utf-8")


def extract_goal(program_text: str) -> str:
    for line in program_text.splitlines():
        if line.startswith("GOAL:"):
            return line.removeprefix("GOAL:").strip()
    return "no explicit GOAL line"


def config_signature(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


if __name__ == "__main__":
    main()
