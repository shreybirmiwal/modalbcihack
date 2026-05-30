# modalbcihack

Hackathon prototype for an autoresearch system that searches Alchemiac
EEG-to-action pipelines. It follows the Karpathy-style loop from
`autoresearch-engine.md`:

- `program.md` is the human steering surface.
- `prepare.py` owns synthetic EEG logs, production CSV loading, session splits,
  nested CV, held-out scoring, sealed-test probes, and the closed-loop metric.
- `pipeline.py` is the agent-editable EEG -> action stack.
- `loop.py` proposes batches of candidate pipelines, evaluates them in parallel,
  and ratchets only improvements that pass a generalization-gap guard.

The implementation is dependency-light for demo reliability: it uses only the
Python standard library.

## Actions

Stage 4 is the production target:

- `nothing`
- `left_squeeze`
- `right_squeeze`
- `eye_blink`

Production CSVs use the Alchemiac header:

```text
timestamp,AF8,AF7,CHEEK_R,CHEEK_L,EAR_R,AFz,BROW_L,NOSE,marker
```

For action-specific recordings, `marker=1` means the action named by the file
happened. Every `marker=0` window in every recording is used as `nothing`; you
do not need to record or provide separate nothing files. Filename hints
currently map positive markers:

- `left*.csv` -> `left_squeeze`
- `right*.csv` -> `right_squeeze`
- `blink*.csv` or `eye*.csv` -> `eye_blink`

## Run

```bash
python3 loop.py --subject S03 --stage 2 --rounds 3 --batch-size 10 --workers 4
```

To run against recorded Alchemiac EEG CSVs instead of the synthetic harness
data, pass repeated `--data-path` values or a `--data-glob`.

```bash
python3 loop.py --subject S03 --stage 4 --rounds 1 --batch-size 4 --workers 2 --data-glob "../bci-sdk/data/*_prod*.csv"
```

Outputs are written to:

- `runs/research_log.jsonl`: every hypothesis, config, metric, and result.
- `runs/best_pipeline_config.json`: the current ratcheted best.
- `runs/final_model.json`: exported final model for inference.

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

Recorded CSVs can be used with the Modal runner too. The local entrypoint
parses and windows the CSVs, sends the compact dataset payload to remote CPU
workers, and exports a final trained model:

```bash
modal run modal_app.py --subject S03 --stage 4 --rounds 3 --batch-size 16 --max-workers 16 --data-glob "../bci-sdk/data/*_prod*.csv" --export-final
```

The Modal path follows the hackathon distributed-compute shape:

- candidate evaluation is embarrassingly parallel through `Function.starmap()`;
- each worker is CPU-only, with `cpu=(0.25, 1.0)` and `memory=(512, 1024)`;
- `max_containers` caps fan-out so a demo cannot accidentally run unbounded;
- remote workers are pure evaluators and do not mutate logs or the ratchet state;
- the local entrypoint keeps `runs/research_log.jsonl` and
  `runs/best_pipeline_config.json` as the experiment notebook.

## Final Model Inference

After Modal training, run batch inference on any Alchemiac EEG CSV:

```bash
python3 infer.py --model runs/final_model.json --input-csv "../bci-sdk/data/blink_prod.csv" --output-csv runs/blink_prod_predictions.csv
```

The output CSV contains one row per causal 32-sample window:

```text
window_index,start_sample,predicted_label,predicted_action
```

For live inference, feed the latest 32 samples from the eight EEG channels in
the same order as the training CSV, build a channel-major `window[8][32]`, and
call `pipeline.predict()` with the model loaded by `pipeline.model_from_payload`.
The exported action names live in `runs/final_model.json` under `actions`.

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
