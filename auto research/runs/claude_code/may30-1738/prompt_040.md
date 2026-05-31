# BCI Claude Code Autoresearch

You are the local Claude Code research agent for this repository, following the
Karpathy/autoresearch pattern.

Read these files before editing:
- README.md
- program.md
- prepare.py
- pipeline.py
- train.py

Rules:
- Edit ONLY `pipeline.py`.
- Do NOT edit `prepare.py`; it is the fixed evaluation/data harness.
- Do NOT install dependencies.
- Keep the action-channel constraint:
  - left_squeeze uses AF7 only (channel index 1)
  - right_squeeze uses AF8 only (channel index 0)
  - eye_blink uses CHEEK_R only (channel index 2)
  - nothing is fallback.
- Optimize reward. Higher is better.
- Current best reward before this iteration: 1.450252

Experiment command:
```bash
uv run python train.py --subject S03 --stage 4 --sealed --export-final --data-glob ../bci-sdk/data/*_prod*.csv > run.log 2>&1
```

Your task for iteration 40:
1. Propose one concrete research idea.
2. Modify `pipeline.py` to implement it.
3. Run the experiment command above.
4. Inspect `run.log`, especially the `reward:` line.
5. Leave `pipeline.py` in the best state you found for this one idea.

The wrapper script will independently re-run evaluation and keep/discard your
change. It logs to `/Users/shreybirmiwal/projects/modalhack/auto research/runs/claude_code/may30-1738/results.tsv`.
