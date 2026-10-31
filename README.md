# scatterbrain-steering

When a system speaks, two jobs get mixed together: **remembering the sentence so far** and **deciding which next word is allowed**. Most language models do both in one stack. If you also want a plan, a value, or a mood-like signal to matter *while* the sentence is being written, you need a place for that signal to land.

This project is that place. A small recurrent language model (a GRU) keeps the unfolding reply in memory. At every word, an **executive poll** reads the turn’s plan, guidance, and chemical levels and adds a bounded nudge to the scores of candidate words:

```text
next-word scores = model scores + (strength × executive nudge)
```

The intended purpose is **controllable generation**: the model still produces the sentence; the executive steers *which* words are more likely, step by step. The data, tests, and gates in this repository exist so that steering can be measured instead of assumed.

This extract is Track C of Scatterbrain. Everyday chat in the parent project still uses a separate speech stack unless you turn Track C on there.

## What works

- **Influence contract.** Polls serialize to JSON and round-trip. Plan coverage only advances when the matching word actually appears. Tests cover determinism, cache across tokens, and a small overfit smoke run (**15** tests pass).
- **Curriculum.** A local teacher produces question/answer/plan triples grounded in source text. Bad teacher output is rejected, not kept. Version 1 of the dataset has **39** approved trajectories; failed lessons stay in the manifest (`artifacts/conscience_curriculum_v1/`).
- **Ablation path.** The same checkpoint can run with executive influence on or off, so you can see whether the nudge moves the output.
- **Portable demo.** A short NumPy script shows biased vs unbiased greedy choice without a full model ([`scripts/demo_executive_bias_numpy.py`](scripts/demo_executive_bias_numpy.py)). Mapping the same idea onto a local GGUF decoder is described in [`docs/GGUF_LOGIT_BIAS_DEMO.md`](docs/GGUF_LOGIT_BIAS_DEMO.md).

Architecture write-up: [`docs/STEERING.md`](docs/STEERING.md). Gate numbers: [`SPEECH_STATUS.md`](SPEECH_STATUS.md).

## What still needs work

- **Fluency.** The last smoke-trained articulator (200 steps, small vocabulary) failed release gates: fluency at 0, repetition on held-out prompts, production flag off. Influence *does* change scores (alignment delta +0.1125); the model is not yet a usable speaker.
- **Learned executive.** The poller today is a designed rule over guidance and chemicals, not a trained network that read the conscience dataset.
- **Default speech.** In the full Scatterbrain repo, Track C stays behind `SCATTERBRAIN_INFLUENCED_ARTICULATOR=1` until those gates pass.

## Try the tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -q
```

Regenerate curriculum from the Scatterbrain monorepo with `run_conscience_curriculum.py` (needs a local Ollama teacher unless you use the test doubles). Parent project: [kmfarris/scatterbrain](https://github.com/kmfarris/scatterbrain).
