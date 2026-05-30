"""Train and export the current best autoresearch model locally."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

from loop import RUNS_DIR, data_path_label, dataset_signature, load_best, resolve_data_specs
from pipeline import model_to_payload, train
from prepare import load_dataset


FINAL_MODEL_PATH = RUNS_DIR / "final_model.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the best discovered pipeline and export an inference artifact.")
    parser.add_argument("--subject", default="S03")
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--data-path", action="append", default=[], help="Alchemiac EEG CSV. Repeat for multiple files.")
    parser.add_argument("--data-glob", action="append", default=[], help="Glob for labeled production CSVs.")
    parser.add_argument("--output", default=str(FINAL_MODEL_PATH))
    args = parser.parse_args()

    RUNS_DIR.mkdir(exist_ok=True)
    data_paths = resolve_data_specs(args.data_path, args.data_glob)
    if not data_paths:
        raise SystemExit("Provide at least one --data-path or --data-glob for real-data final training.")

    dataset_id = dataset_signature(args.subject, args.stage, data_paths)
    best_config, best_reward = load_best(args.subject, args.stage, data_paths)
    dataset = load_dataset(subject=args.subject, stage=args.stage, data_path=data_paths)
    model = train(dataset.examples, len(dataset.actions), best_config)
    label_counts = Counter(example.label for example in dataset.examples)

    artifact = {
        "subject": args.subject,
        "stage": args.stage,
        "actions": dataset.actions,
        "dataset_id": dataset_id,
        "data_path": data_path_label(data_paths),
        "best_reward": best_reward,
        "config": best_config,
        "model": model_to_payload(model),
        "label_counts": {dataset.actions[label]: count for label, count in sorted(label_counts.items())},
        "window": {"channels": 8, "samples": 32, "sample_rate_hz": 128},
        "backend": "local",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {output}")
    print(json.dumps({"dataset_id": dataset_id, "best_reward": best_reward, "label_counts": artifact["label_counts"]}, indent=2))


if __name__ == "__main__":
    main()
