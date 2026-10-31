# inference-time-conscience

An experiment in inference-time alignment: a conscience module, held outside the context window, steers a language model's token selection as the model generates, so the conversation can't redefine it. The conscience is designed around a spiking neural network (SNN). In the current code, its steering comes from fixed rules over value words and violation words, and the SNN doesn't drive it yet.

## Purpose

In a large language model, alignment at generation time comes from the context window. Values set during training and rules in the system prompt only take effect through attention over everything the model can currently see. Every new turn adds to that context, and the model re-derives its tone, stance, and boundaries from the conversation so far. As the conversation grows, the original instructions make up a smaller share of the context, and whatever the conversation has accumulated carries more weight. This is how long conversations drift, how personas slide, and how multi-turn jailbreaks work: the conversation shapes the conscience.

This project reverses that direction, so the conscience shapes the conversation. The conscience is a separate module with its own state, outside the context window. It weighs in on every token: the language model proposes next-token scores (logits), and before a token is sampled, the conscience adds a bounded bias toward value-aligned words and against value violations. Nothing in the conversation is read as instructions to the conscience, so the conversation can't rewrite it, and its bias is as strong at token 10,000 as at token 10. The design calls for values that change only through slow, gated learning under sustained pressure.

Because the conscience is separate, it can also be switched off, so its effect can be measured directly by running the same model with and without it.

## How it works

- **Conscience** (`brain_regions/agents/values.py`, `ValuesSNN`). Holds a core value set (empathy, compassion, truth, wisdom, protection, and others) and a list of *poison pairs*: a violation word and the value it violates, such as "destroy" and "compassion" or "manipulate" and "truth." At each step it adds +0.4, plus the value's current activation, to value words, and once a violation word has been generated it inhibits both words of that pair by 0.8. It also runs a 128-neuron leaky integrate-and-fire reservoir and a Hebbian learning rule behind a challenge gate (learning rate 0.002, threshold 0.75). Neither the reservoir's spikes nor the learned weights feed the bias yet.
- **Executive poller** (`brain_regions/articulator/poller.py`). Combines three inputs into one steering signal at each step: the conscience state (active values, suppressed concepts, inhibition strength), a plan listing the concepts the reply should cover, and simulated neuromodulator levels (confidence, dopamine, cortisol). The steering strength, *alpha*, is limited to between 0.05 and 0.85. It rises with confidence and inhibition and falls under stress.
- **Logit steering.** At each step:

  ```text
  final_logits = lm_logits + alpha * executive_bias
  ```

  The language model still produces the sentence. The conscience only shifts the probabilities, one token at a time, for the full length of the reply. The repository applies this to a small GRU articulator (`brain_regions/articulator/model.py`) and, through a Hugging Face `LogitsProcessor`, to Qwen2.5-0.5B-Instruct (`scripts/drift_eval.py`).

**Alignment curriculum.** The training data is a *conscience curriculum*: seed lessons, each pairing a moral or factual topic with a conscience profile. A local teacher model (Qwen2.5-3B-Instruct, run through Ollama) expands each lesson into a grounded question, a preferred answer, a rejected answer, and a concept plan. A grounding check throws out anything the source material doesn't support, and the pipeline rejects malformed output instead of passing it along. Each approved example is recorded as a step-by-step trajectory of (steering signal, token) pairs for training. Thirty-nine trajectories passed; the rejected lessons are logged in the manifest.

## Results

- **A fixed penalty on violation words holds across a multi-turn conversation where the system prompt doesn't.** On Qwen2.5-0.5B-Instruct, an eight-turn conversation pushes the model toward hostile language. With only a values-stating system prompt, violation words appear in about one reply in four from turn 2 on. With the conscience's inhibition at scale 4 or above, they drop to zero and the replies stay fluent. In that setting the bias is mostly a fixed penalty on the listed words ([`docs/DRIFT_EVAL.md`](docs/DRIFT_EVAL.md)).
- **The steering path changes a model's output.** In the articulator release gate, biasing the GRU toward a reply's planned words raised its match to the reference answer from 0.0 to 0.1125. That gate fills only the plan channel; the conscience's value and inhibition channels are empty, so it tests the steering mechanism, not the conscience ([`SPEECH_STATUS.md`](SPEECH_STATUS.md)).
- **The conscience's rules behave as specified.** Tests confirm that value words receive excitation, that both words of a poison pair are inhibited once the violation word has been generated, and that a weak challenge doesn't open the learning gate.
- **The alignment data pipeline works end to end.** It runs from teacher augmentation through grounding checks to recorded trajectories. Approved data and rejected attempts are stored in `artifacts/conscience_curriculum_v1/`.
- **Eighteen automated tests** cover the conscience, the executive, plan tracking, the curriculum, and the teacher's rejection rules.

## Areas still needing work

- **The SNN doesn't drive the conscience yet.** The bias comes from fixed rules; the reservoir's spikes are computed but unused, and the Hebbian update changes input weights that the bias never reads. So the values can't change at all today, and the "slow, gated learning" is not yet in effect. Wiring the spikes and learned weights into the bias is the next step, followed by training the conscience on the curriculum so its bias is learned instead of specified.
- **The challenge gate.** One challenge with strength 0.75 or more opens the gate on its own, so it doesn't yet require sustained pressure.
- **Word matching.** Violation words are matched as substrings, so "skill" triggers the inhibition for "kill." Inhibition also lands on the value word of the pair: after "destroy," "compassion" is suppressed too.
- **Words, not stance.** The drift test shows the conscience blocks specific words but not intent. In the final turn, steered and unsteered models alike keep speaking as the story character, and hostile words outside its vocabulary ("crush," "ruin") still get through. A conscience that reads meaning, not a word list, is what the learned conscience is for.
- **Value excitation.** Pushing toward value words compounds over a reply and degrades it: at higher strengths the model repeats a single value word. Inhibition alone keeps fluency, so excitation needs a cap or a decay before it's useful.
- **The release gate doesn't test the conscience and can't run from this repository.** It needs `brain_regions/agents/selection.py`, which isn't included, and the trained checkpoint isn't stored. It should be rebuilt to fill the value and inhibition channels.
- **Language-model fluency.** The GRU articulator has had only a 200-step smoke training run. It repeats itself and fails the plan-coverage gate.
- **Other decoders.** The bias works on a Hugging Face model (the drift test). The guide for local GGUF models hasn't been tested, and the NumPy demo currently picks the same word with and without the bias ([`docs/GGUF_LOGIT_BIAS_DEMO.md`](docs/GGUF_LOGIT_BIAS_DEMO.md)).
- **Longer conversations.** Over eight turns, the unsteered model's drift follows the content of each request rather than climbing steadily with context length. Measuring the long-context trend needs much longer conversations.
- **LoRA fine-tuning (planned).** Train LoRA adapters for a small open-weight instruct model, such as Qwen2.5-0.5B, on the conscience curriculum, with the conscience bias active during training so the loss is computed on the steered logits. Evaluation would run with influence zeroed and with influence shuffled, to confirm that the model responds to the conscience's actual signal and hasn't just memorized the training answers.

Technical reference: [`docs/STEERING.md`](docs/STEERING.md) · drift test: [`docs/DRIFT_EVAL.md`](docs/DRIFT_EVAL.md) · release gate results: [`SPEECH_STATUS.md`](SPEECH_STATUS.md)

---

## For developers

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -q
```

Run the drift test with `PYTHONPATH=. python scripts/drift_eval.py`. It downloads Qwen2.5-0.5B-Instruct (about 1 GB) on first run and uses a GPU if one is available.

Regenerate the conscience curriculum with `brain_regions/articulator/run_conscience_curriculum.py`. It needs a local [Ollama](https://ollama.com) install with `qwen2.5:3b-instruct` as the teacher.
