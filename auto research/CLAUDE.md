# Claude Code Autoresearch Instructions

This repo follows the Karpathy/autoresearch pattern for BCI action decoding.

## Files

- `prepare.py` is the fixed data/evaluation harness. Do not edit it.
- `pipeline.py` is the only agent-editable model/training file.
- `train.py` is the single-experiment metric entrypoint.
- `program.md` is the human steering document.

## Goal

Improve `reward` from:

```bash
uv run python train.py --subject S03 --stage 4 --sealed --export-final --data-glob "../bci-sdk/data/*_prod*.csv"
```

Higher `reward` is better. The run also exports `runs/final_model.json`, which
the live BCI game controller auto-reloads when it changes.

## Hard Constraints

Preserve channel isolation:

- `left_squeeze` may only inspect `AF7` (`channel index 1`).
- `right_squeeze` may only inspect `AF8` (`channel index 0`).
- `eye_blink` may only inspect `CHEEK_R` (`channel index 2`).
- `nothing` is the fallback when no action detector fires.

Do not add dependencies. Keep inference lightweight enough to run live on the
MacBook.

## Run The Local Agent Wrapper

From `auto research/`:

```bash
uv run python claude_research.py --iterations 25 --max-budget-usd 25 --data-glob "../bci-sdk/data/*_prod*.csv"
```

The wrapper asks local Claude Code to edit `pipeline.py`, reruns `train.py`, and
keeps or discards the edit based on reward. Results are written under
`runs/claude_code/<tag>/results.tsv` and are surfaced in the dashboard.
