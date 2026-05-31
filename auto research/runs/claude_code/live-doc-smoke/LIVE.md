# Claude Code Autoresearch Live Log

- started: `2026-05-30 17:36:25`
- run_dir: `/Users/shreybirmiwal/projects/modalhack/auto research/runs/claude_code/live-doc-smoke`
- iterations: `1`
- max_budget_usd: `3.0`
- claude_output_format: `stream-json`
- data_glob: `['../bci-sdk/data/*_prod*.csv']`
- data_path: `[]`


### Baseline evaluation

```text
$ uv run python train.py --subject S03 --stage 4 --sealed --export-final --data-glob ../bci-sdk/data/*_prod*.csv
---
reward:             1.406824
heldout_score:      1.332111
balanced_accuracy:  0.249639
macro_f1:           0.247851
generalization_gap: 0.000000
training_seconds:   5.6
dataset_id:         bff1f19f7ffc2eb6
stage:              4
final_model_path:   runs/final_model.json
```

## Baseline

- reward: `1.406824`
- train entrypoint: `uv run python train.py --subject S03 --stage 4 --sealed --export-final --data-glob ../bci-sdk/data/*_prod*.csv`

## Iteration 1

Dry run wrote `prompt_001.md`; Claude was not invoked.
