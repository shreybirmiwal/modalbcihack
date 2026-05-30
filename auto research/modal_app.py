"""Modal CPU fan-out for the EEG autoresearch demo.

Run with:
    modal run modal_app.py --rounds 3 --batch-size 32 --max-workers 32
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import time

import modal

from loop import (
    append_log,
    config_signature,
    extract_goal,
    load_best,
    load_program_text,
    load_tried_signatures,
    propose_batch,
    save_best,
)


app = modal.App("bci-autoresearch-cpu")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_python_source("loop")
    .add_local_python_source("prepare")
    .add_local_python_source("pipeline")
)


@app.function(
    image=image,
    cpu=(0.25, 1.0),
    memory=(512, 1024),
    timeout=300,
    retries=1,
    max_containers=64,
    scaledown_window=60,
)
def evaluate_candidate_remote(
    candidate_payload: dict,
    subject: str,
    stage: int,
    sealed: bool,
    program_hash: str,
    experiments_seen: int,
) -> dict:
    from prepare import evaluate_config

    started = time.time()
    config = candidate_payload["config"]
    evaluation = evaluate_config(config, subject=subject, stage=stage, sealed=sealed)
    multiple_comparison_penalty = 0.006 * math.log1p(max(1, experiments_seen))
    reward = evaluation.reward_base - 0.55 * evaluation.generalization_gap - multiple_comparison_penalty
    return {
        "timestamp": round(time.time(), 3),
        "subject": subject,
        "stage": stage,
        "program_hash": program_hash,
        "signature": candidate_payload["signature"],
        "hypothesis": candidate_payload["hypothesis"],
        "config": config,
        "cv": asdict(evaluation.cv),
        "heldout": asdict(evaluation.heldout),
        "sealed": asdict(evaluation.sealed) if evaluation.sealed else None,
        "generalization_gap": evaluation.generalization_gap,
        "reward": reward,
        "seconds": round(time.time() - started, 3),
        "backend": "modal-cpu",
    }


@app.local_entrypoint()
def main(
    subject: str = "S03",
    stage: int = 2,
    rounds: int = 3,
    batch_size: int = 32,
    max_workers: int = 32,
    sealed_probe_every: int = 3,
    gap_tolerance: float = 0.18,
) -> None:
    """Run the ratchet locally while candidate scoring fans out on Modal CPUs."""

    program_text = load_program_text()
    program_hash = hashlib.sha256(program_text.encode("utf-8")).hexdigest()[:12]
    tried = load_tried_signatures()
    best_config, best_reward = load_best(subject, stage)

    print(f"Modal autoresearch EEG loop | subject={subject} stage={stage}")
    print(f"program=program.md hash={program_hash} goal={extract_goal(program_text)}")
    print(
        "remote_cpu="
        "request=0.25 limit=1.0 memory=512-1024MiB "
        f"max_containers={min(max_workers, 64)}"
    )
    print(f"starting_best_reward={best_reward:.4f} tried={len(tried)}")

    remote_function = evaluate_candidate_remote.with_options(max_containers=min(max_workers, 64))

    for round_idx in range(1, rounds + 1):
        candidates = propose_batch(
            base_config=best_config,
            tried=tried,
            batch_size=batch_size,
            round_idx=round_idx,
            subject=subject,
            stage=stage,
        )
        candidate_payloads = [
            {
                "config": candidate.config,
                "hypothesis": candidate.hypothesis,
                "signature": candidate.signature or config_signature(candidate.config),
            }
            for candidate in candidates
        ]
        sealed = sealed_probe_every > 0 and round_idx % sealed_probe_every == 0
        print(f"\nround={round_idx} candidates={len(candidate_payloads)} sealed_probe={sealed}")

        inputs = [
            (payload, subject, stage, sealed, program_hash, len(tried) + idx + 1)
            for idx, payload in enumerate(candidate_payloads)
        ]
        results = list(
            remote_function.starmap(
                inputs,
                order_outputs=False,
                return_exceptions=False,
            )
        )

        accepted = False
        for result in sorted(results, key=lambda row: row["reward"], reverse=True):
            tried.add(result["signature"])
            append_log(result)
            if result["reward"] > best_reward and result["generalization_gap"] <= gap_tolerance:
                best_reward = result["reward"]
                best_config = result["config"]
                save_best(best_config, best_reward, subject, stage, result)
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

    print("\nBest Modal-backed config:")
    print(json.dumps({"reward": best_reward, "config": best_config}, indent=2, sort_keys=True))
