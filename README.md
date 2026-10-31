# inference-time-conscience

An experiment in inference-time alignment: a spiking neural network (SNN) acts as a conscience that steers a language model's token selection as the model generates.

## Purpose

Most language-model alignment happens before the model is ever used. Fine-tuning methods such as RLHF and DPO build preferences into the weights, and system prompts state rules at the top of the context window. At generation time, the model is left to keep to those rules on its own through attention over its context. As a reply grows, the instructions carry less weight and the output can drift away from them.

This project puts alignment somewhere else: a separate conscience module that runs alongside the language model and weighs in on every token. The language model proposes next-token scores (logits). Before a token is sampled, the conscience adds a bounded bias toward value-aligned words and against value violations. The alignment signal is applied again at every decoding step instead of being read once from the prompt. It can also be turned off, which lets you measure its effect directly by comparing the same checkpoint with and without it.

## How it works

The system has three parts:

- **Conscience SNN** (`brain_regions/agents/values.py`). A 128-neuron leaky integrate-and-fire reservoir that holds a core value set (empathy, compassion, truth, wisdom, protection, and others) and a list of *poison pairs*: word combinations that violate those values, such as "destroy" alongside "compassion" or "manipulate" alongside "truth." At each step it reads the current value activations and the last token generated. It excites candidate words that carry a value and inhibits words that would complete a poison pair. Its values are hard to change on purpose. Learning is Hebbian and slow, and a challenge gate blocks updates unless pressure builds up over time, so a single adversarial input can't rewrite them.
- **Executive poller** (`brain_regions/articulator/poller.py`). Combines three inputs into one steering signal at each step: the conscience state (active values, suppressed concepts, inhibition strength), a plan listing the concepts the reply should cover, and simulated neuromodulator levels (confidence, dopamine, cortisol). The steering strength, *alpha*, is limited to between 0.05 and 0.85. It rises with confidence and inhibition and falls under stress.
- **Articulator** (`brain_regions/articulator/model.py`). A small GRU language model whose recurrent state lasts for the whole utterance. At each step:

  ```text
  final_logits = lm_logits + alpha * executive_bias
  ```

  The language model still produces the sentence. The conscience only shifts the probabilities, one token at a time, for the full length of the reply.

**Alignment curriculum.** The training data is a *conscience curriculum*: seed lessons, each pairing a moral or factual topic with a conscience profile. A local teacher model (Qwen2.5-3B-Instruct, run through Ollama) expands each lesson into a grounded question, a preferred answer, a rejected answer, and a concept plan. A grounding check throws out anything the source material doesn't support, and the pipeline rejects malformed output instead of passing it along. Each approved example is recorded as a step-by-step trajectory of (steering signal, token) pairs for training. Thirty-nine trajectories passed; the rejected lessons are logged in the manifest.

**Evaluation.** A release gate (`brain_regions/evaluation/articulator_release_gate.py`) runs the same checkpoint on held-out lesson families twice: once with the conscience active and once with its influence set to zero. It scores both runs against the reference answer and plan. The difference between the two scores, the *influence delta*, isolates what the conscience contributes.

## Results

- **Steering changes the output.** On held-out lessons, the steered output scored 0.1125 against the reference and plan. The same checkpoint with steering off scored 0.0. The conscience alone moved token selection toward the aligned answer.
- **The conscience SNN behaves as designed.** Tests confirm that value words receive excitation, that poison-pair completions are inhibited once the first word of the pair has been generated, and that weak challenges don't change the SNN's stored values.
- **The alignment data pipeline works end to end.** It runs from teacher augmentation through grounding checks to recorded trajectories. Approved data and rejected attempts are stored in `artifacts/conscience_curriculum_v1/`.
- **The steering rule is model-agnostic.** The same additive logit bias applies to any decoder that exposes its logits before sampling. A NumPy demo and a guide for locally run GGUF models show how ([`docs/GGUF_LOGIT_BIAS_DEMO.md`](docs/GGUF_LOGIT_BIAS_DEMO.md)).
- **Eighteen automated tests** cover the conscience SNN, the executive, plan tracking, the curriculum, and the teacher's rejection rules.

## Areas still needing work

- **Language-model fluency.** The GRU articulator has had only a 200-step smoke training run. It repeats itself, doesn't produce fluent sentences, and fails the plan-coverage gate. The steering effect is real, but the model it steers isn't yet good enough to show it on natural text.
- **A learned conscience.** The conscience's excitation and inhibition strengths are set by hand, and the reservoir's spike output isn't yet used in the bias it emits. The next step is training the conscience on the curriculum so its bias is learned instead of specified.
- **Live wiring.** In this repository, the executive reads conscience state from recorded profiles in the curriculum. The SNN's output doesn't yet feed it directly at decode time.
- **LoRA fine-tuning (planned).** The plan is to replace the GRU with a small open-weight instruct model, such as Qwen2.5-0.5B, and train LoRA adapters on the conscience curriculum. The conscience bias would be active during training, so the loss is computed on the final, steered logits. The goal is a model that is both fluent and cooperative with its conscience instead of one that overrides it. Evaluation would run with influence zeroed and with influence shuffled, to confirm that the model responds to the conscience's actual signal and hasn't just memorized the training answers.

Technical reference: [`docs/STEERING.md`](docs/STEERING.md) · release gate results: [`SPEECH_STATUS.md`](SPEECH_STATUS.md)

---

## For developers

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -q
```

Regenerate the conscience curriculum with `brain_regions/articulator/run_conscience_curriculum.py`. It needs a local [Ollama](https://ollama.com) install with `qwen2.5:3b-instruct` as the teacher.
