# scatterbrain-steering

Track C decoder: a GRU holds state across the utterance; each step an executive poll adds bounded bias to vocabulary logits. Plan coverage, chemical levels, and structured guidance feed the poll—they do not replace the LM.

`final_logits = lm_logits + α · executive_bias` — see [`docs/STEERING.md`](docs/STEERING.md).

## Tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -q
```

## Data

`artifacts/conscience_curriculum_v1/` — trajectories, teacher cache entries, manifest including failed lessons. Regenerate from the Scatterbrain monorepo with `run_conscience_curriculum.py`.

## Upstream

Production speech and the full mesh: [kmfarris/scatterbrain](https://github.com/kmfarris/scatterbrain). [`SPEECH_STATUS.md`](SPEECH_STATUS.md) describes which path runs by default.

Maintainers refresh this tree with `utero_ai/scripts/export_scatterbrain_steering.sh`.
