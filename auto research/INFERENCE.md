# Inference Guide

The latest Modal-trained model is exported to:

```text
runs/final_model.json
```

It predicts four actions:

- `nothing`
- `left_squeeze`
- `right_squeeze`
- `eye_blink`

## Batch Inference

From `auto research/`:

```bash
python3 infer.py --model runs/final_model.json --input-csv "../bci-sdk/data/blink_prod.csv" --output-csv runs/blink_prod_predictions.csv
```

The output file has:

```text
window_index,start_sample,predicted_label,predicted_action
```

Each row is one causal 32-sample EEG window. `start_sample` is the first source
sample in that window.

## Live Inference

Load the exported artifact and reuse the pipeline helpers:

```python
import json
from pipeline import model_from_payload, predict
from prepare import Example

artifact = json.loads(open("runs/final_model.json").read())
actions = artifact["actions"]
model = model_from_payload(artifact["model"])

# window must be channel-major: 8 channels x 32 samples.
example = Example(session=0, frame=0, window=window, label=0)
label = predict(model, [example])[0]
action = actions[label]
```

Use the same channel order as the Alchemiac CSV header:

```text
AF8, AF7, CHEEK_R, CHEEK_L, EAR_R, AFz, BROW_L, NOSE
```

## Retraining With New Files

Drop new labeled recordings into `bci-sdk/data/` with names like:

```text
left_prod3.csv
right_prod2.csv
blink_prod3.csv
```

No separate `nothing` file is required. Any window whose source samples have
`marker=0` is trained as `nothing`; positive markers are assigned to the action
implied by the filename.

Then rerun Modal:

```bash
modal run modal_app.py --subject S03 --stage 4 --rounds 3 --batch-size 16 --max-workers 16 --data-glob "../bci-sdk/data/*_prod*.csv" --export-final
```

The dashboard reads `runs/research_log.jsonl` and
`runs/best_pipeline_config.json`, so it will update after the run finishes.
