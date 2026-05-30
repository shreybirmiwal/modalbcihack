# modalbcihack

Hackathon prototype for an autoresearch system that searches EEG-to-game-control
pipelines. It follows the Karpathy-style loop from `autoresearch-engine.md`:

- `program.md` is the human steering surface.
- `prepare.py` is frozen and owns synthetic EEG logs, session splits, nested CV,
  held-out scoring, sealed-test probes, and the closed-loop game metric.
- `pipeline.py` is the agent-editable EEG -> action stack.
- `loop.py` proposes batches of candidate pipelines, evaluates them in parallel,
  and ratchets only improvements that pass a generalization-gap guard.

The implementation is dependency-light for demo reliability: it uses only the
Python standard library.

## Run

```bash
python3 loop.py --subject S03 --stage 2 --rounds 3 --batch-size 10 --workers 4
```

Outputs are written to:

- `runs/research_log.jsonl`: every hypothesis, config, metric, and result.
- `runs/best_pipeline_config.json`: the current ratcheted best.

Try harder curricula:

```bash
python3 loop.py --subject S03 --stage 3 --rounds 5 --batch-size 16 --workers 4
python3 loop.py --subject S03 --stage 4 --rounds 5 --batch-size 16 --workers 4
```

## Run on Modal CPUs

Install and authenticate the Modal CLI once:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
modal setup
```

Then fan out candidate evaluation across CPU containers:

```bash
modal profile activate hackathon-modal-sf
modal run modal_app.py --subject S03 --stage 2 --rounds 3 --batch-size 32 --max-workers 32
```

The Modal path follows the hackathon distributed-compute shape:

- candidate evaluation is embarrassingly parallel through `Function.starmap()`;
- each worker is CPU-only, with `cpu=(0.25, 1.0)` and `memory=(512, 1024)`;
- `max_containers` caps fan-out so a demo cannot accidentally run unbounded;
- remote workers are pure evaluators and do not mutate logs or the ratchet state;
- the local entrypoint keeps `runs/research_log.jsonl` and
  `runs/best_pipeline_config.json` as the experiment notebook.

## Live Dashboard

The React dashboard watches `runs/research_log.jsonl` and
`runs/best_pipeline_config.json` through a small Vite API. It updates while
local or Modal-backed runs are writing new candidates.

```bash
cd ../web
npm install
npm run dev
```

Open the printed local URL. By default Vite uses:

```text
http://127.0.0.1:5173
```

## Demo Story

The core pitch is that a human should not hand-search the space of EEG
preprocessing, windowing, channel selection, features, classifiers, and temporal
smoothing. The engine does that as an autonomous research loop:

1. Read `program.md` for goals and constraints.
2. Propose many candidate edits with hypotheses.
3. Fan out evaluation jobs.
4. Score with nested CV plus held-out closed-loop game performance.
5. Accept only the best improvement.
6. Keep a structured research notebook for future rounds.
