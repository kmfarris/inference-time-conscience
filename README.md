# scatterbrain-steering: steering what an AI says while it's speaking

## What it is

An experiment in guiding an AI's reply word by word, while the reply is being written.

## The problem

An AI that generates text picks each next word based on the words it has already written. If you want a goal to shape the whole reply, such as "mention these three points," "stay gentle," or "avoid this topic," you usually state it once at the start and hope the model sticks with it. Partway through, it often drifts.

## How it works

Two parts work together:

- **The writer.** A small language model writes the reply one word at a time and keeps a running memory of everything it has written so far.
- **The executive.** Before each word, a separate component looks at the plan for the reply (which points still need covering) and the system's current state (such as how calm or stressed it is). It gives a small push toward words that fit the plan and away from words that don't.

The writer still produces the sentence. The executive only shifts the odds, one word at a time, for the whole length of the reply. Because the push can be turned off, it's easy to measure how much it changes the output.

**Training examples.** A local AI "teacher" was given short source texts and asked to write questions, good answers, deliberately bad answers, and a plan for each answer. Any output that broke the rules, such as answers that didn't stick to the source, was thrown out automatically. Thirty-nine examples passed, and the rejected ones are recorded too.

## Results so far

- **The steering works.** Turning the executive's push on or off measurably changes which words get chosen.
- **It's tested.** Fifteen automated tests cover the executive, the plan tracking, the training examples, and the teacher's rule checks.
- **The examples are saved.** The 39 accepted examples and the rejected attempts are stored in `artifacts/conscience_curriculum_v1/`.
- **It ports to other models.** A short demo shows the same push applied to a model's word scores, and a guide explains how to do it with locally run open-source models ([`docs/GGUF_LOGIT_BIAS_DEMO.md`](docs/GGUF_LOGIT_BIAS_DEMO.md)).

## Still to do

- **Fluency.** The small language model has only had a short training run and can't write fluent sentences yet. It tends to repeat itself, so it fails its quality checks.
- **Learning the executive.** The executive currently follows rules written by hand. The next step is having it learn from the training examples.
- **Switched off by default.** This project is one part of a larger system, Scatterbrain. There, everyday conversation uses a different speech method, and this one stays switched off until it passes its quality checks.

Technical details: [`docs/STEERING.md`](docs/STEERING.md) · quality checks: [`SPEECH_STATUS.md`](SPEECH_STATUS.md)

---

## For developers

Run the tests:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -q
```

The core formula: each next-word score = the language model's score + (strength × the executive's push).

The training examples can be regenerated from the main Scatterbrain project with `run_conscience_curriculum.py`, which needs a local [Ollama](https://ollama.com) model as the teacher. Main project: [kmfarris/scatterbrain](https://github.com/kmfarris/scatterbrain).
