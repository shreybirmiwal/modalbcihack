"""Single-experiment entrypoint for Claude-driven autoresearch.

This is the BCI equivalent of Karpathy autoresearch's `train.py`: Claude edits
`pipeline.py`, then this script evaluates that exact pipeline against the fixed
`prepare.py` harness and prints a compact metric block.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import time

from loop import RUNS_DIR, data_path_label, dataset_signature, resolve_data_specs
from pipeline import CONFIG, model_to_payload, train as train_model
from prepare import evaluate_config, load_dataset


FINAL_MODEL_PATH = RUNS_DIR / "final_model.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current pipeline.py for one autoresearch experiment.")
    parser.add_argument("--subject", default="S03")
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--data-path", action="append", default=[], help="Alchemiac EEG CSV. Repeat for multiple files.")
    parser.add_argument("--data-glob", action="append", default=[], help="Glob for labeled production CSVs.")
    parser.add_argument("--sealed", action="store_true", help="Also evaluate sealed session.")
    parser.add_argument("--export-final", action="store_true", help="Train/export final_model.json using current pipeline.py.")
    args = parser.parse_args()

    RUNS_DIR.mkdir(exist_ok=True)
    started = time.time()
    data_paths = resolve_data_specs(args.data_path, args.data_glob)
    dataset_id = dataset_signature(args.subject, args.stage, data_paths)
    evaluation = evaluate_config(CONFIG, subject=args.subject, stage=args.stage, sealed=args.sealed, data_path=data_paths)
    reward = evaluation.reward_base - 0.55 * evaluation.generalization_gap
    seconds = time.time() - started

    print("---")
    print(f"reward:             {reward:.6f}")
    print(f"heldout_score:      {evaluation.heldout.closed_loop_score:.6f}")
    print(f"balanced_accuracy:  {evaluation.heldout.balanced_accuracy:.6f}")
    print(f"macro_f1:           {evaluation.heldout.macro_f1:.6f}")
    print(f"generalization_gap: {evaluation.generalization_gap:.6f}")
    print(f"training_seconds:   {seconds:.1f}")
    print(f"dataset_id:         {dataset_id}")
    print(f"stage:              {args.stage}")

    if args.export_final and data_paths:
        dataset = load_dataset(subject=args.subject, stage=args.stage, data_path=data_paths)
        model = train_model(dataset.examples, len(dataset.actions), CONFIG)
        artifact = {
            "subject": args.subject,
            "stage": args.stage,
            "actions": dataset.actions,
            "dataset_id": dataset_id,
            "data_path": data_path_label(data_paths),
            "reward": reward,
            "config": CONFIG,
            "model": model_to_payload(model),
            "metrics": {
                "cv": asdict(evaluation.cv),
                "heldout": asdict(evaluation.heldout),
                "sealed": asdict(evaluation.sealed) if evaluation.sealed else None,
                "generalization_gap": evaluation.generalization_gap,
            },
            "window": {"channels": 8, "samples": 32, "sample_rate_hz": 128},
            "backend": "local-claude-code",
        }
        FINAL_MODEL_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"final_model_path:   {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
