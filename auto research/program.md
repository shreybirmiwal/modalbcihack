# Subject: S03 | Stage: 2 actions (noop, jump)

GOAL: maximize closed-loop survival time in a small game by decoding causal EEG
windows into controls.

CONSTRAINTS:
- Use causal windows only; never read future frames.
- Keep inference lightweight enough for live feedback.
- Prefer <=8 channels.
- Do not edit `prepare.py`; it owns data generation, splits, and evaluation.

SEARCH PRIORITIES:
1. Window crops and offsets.
2. Motor-rhythm features: mu/beta-style band energies and channel summaries.
3. Artifact robustness.
4. Model family: centroid -> perceptron -> softmax.
5. Temporal smoothing only when it improves closed-loop play.

CURRICULUM:
- Stage 2: noop, jump.
- Stage 3: add left after stable survival improvement.
- Stage 4: add right after stable survival improvement.

NOTES FOR AGENT:
- Every candidate should have a short hypothesis.
- Avoid re-trying configurations already logged in `runs/research_log.jsonl`.
- Accept only candidates that improve reward and pass the generalization-gap guard.
