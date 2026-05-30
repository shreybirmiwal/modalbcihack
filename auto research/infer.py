"""Run batch inference with an exported autoresearch model."""

from __future__ import annotations

from collections import Counter
import argparse
import csv
import json
from pathlib import Path

from pipeline import model_from_payload, predict
from prepare import Example, windows_from_recorded_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict BCI actions for an Alchemiac EEG CSV.")
    parser.add_argument("--model", default="runs/final_model.json")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="runs/predictions.csv")
    args = parser.parse_args()

    artifact = json.loads(Path(args.model).read_text(encoding="utf-8"))
    actions = list(artifact["actions"])
    model = model_from_payload(artifact["model"])
    windows = windows_from_recorded_csv(args.input_csv, len(actions))
    examples = [
        Example(session=0, frame=idx, window=window, label=label)
        for idx, (window, label) in enumerate(windows)
    ]
    predictions = predict(model, examples)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_index", "start_sample", "predicted_label", "predicted_action"])
        for idx, prediction in enumerate(predictions):
            writer.writerow([idx, idx * 32, prediction, actions[prediction]])

    counts = Counter(actions[prediction] for prediction in predictions)
    print(f"wrote {len(predictions)} predictions to {output}")
    print(json.dumps(dict(sorted(counts.items())), indent=2))


if __name__ == "__main__":
    main()
