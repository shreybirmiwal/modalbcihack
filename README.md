# modalbcihack

Autoresearch BCI hackathon prototype: an autonomous research loop that searches
EEG-to-game-control pipelines and a live React dashboard for inspecting loss
improvements, candidate runs, and Modal CPU fan-out.

## Project Layout

- `auto research/`: Python autoresearch engine, frozen eval harness, Modal CPU
  runner, and program instructions.
- `web/`: React/Vite dashboard for visualizing candidate runs and improvement
  ladders.

## Quick Start

```bash
cd "auto research"
python3 loop.py --subject S03 --stage 2 --rounds 3 --batch-size 10 --workers 4
```

```bash
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.
