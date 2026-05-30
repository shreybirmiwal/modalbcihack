# Subject: S03 | Stage: 4 actions (nothing, left_squeeze, right_squeeze, eye_blink)

GOAL: decode causal Alchemiac EEG windows into nothing, left arm squeeze, right arm squeeze, and eye blink actions.

CONSTRAINTS:
- Use causal windows only; never read future frames.
- Keep inference lightweight enough for live feedback.
- Prefer <=8 channels.
- Real production CSVs use marker=1 for the action named by the filename and
  marker=0 for nothing.

SEARCH PRIORITIES:
1. Window crops and offsets.
2. Motor-rhythm features: mu/beta-style band energies and channel summaries.
3. Artifact robustness.
4. Model family: centroid -> perceptron -> softmax.
5. Temporal smoothing only when it improves closed-loop play.

CURRICULUM:
- Stage 2: nothing and left_squeeze.
- Stage 3: add right_squeeze.
- Stage 4: add eye_blink.

NOTES FOR AGENT:
- Every candidate should have a short hypothesis.
- Avoid re-trying configurations already logged in `runs/research_log.jsonl`.
- Accept only candidates that improve reward and pass the generalization-gap guard.
