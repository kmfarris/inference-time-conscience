# scatterbrain-steering

Executive influence on a persistent decoder: turn-level guidance becomes bounded logit bias while a recurrent language model holds decode memory. This repository is the public extract of Track C from the Scatterbrain monorepo—tests, curriculum artifacts, and evaluation gates included.

## Architecture in one pass

Each generated token: GRU step → executive poll → `final_logits = lm_logits + α · bias`. The poll reads plan coverage, chemical levels, and structured guidance; it does not replace the LM or emit text directly. Diagram and scope: [`docs/STEERING.md`](docs/STEERING.md).

## Quick verification

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -q
```

## Curriculum artifact

`artifacts/conscience_curriculum_v1/` — teacher-grounded trajectories; manifest retains failed lessons. Regenerate from the monorepo via `run_conscience_curriculum.py` (local Ollama teacher when not using test fakes).

## Monorepo

Full mesh and production speech: [kmfarris/scatterbrain](https://github.com/kmfarris/scatterbrain). Read [`SPEECH_STATUS.md`](SPEECH_STATUS.md) before equating Track C with default speech.

## Sync

From `utero_ai`: `scripts/export_scatterbrain_steering.sh [/path/to/scatterbrain-steering]`.
