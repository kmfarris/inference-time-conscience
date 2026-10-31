# Logit bias hook for local GGUF decode

Track C already implements residual bias in PyTorch: `final_logits = lm_logits + α · executive_bias`. The same contract ports to llama.cpp-style decode by supplying a sparse bias vector each step—no LoRA required for a before/after demonstration.

## Mapping

| Track C field | GGUF decode hook |
|---------------|------------------|
| `concept_bias[token_id]` | `llama_set_logits_bias` / equivalent per candidate id |
| `alpha` | Scalar multiplier on the bias vector |
| Plan slot coverage | Increase bias mass on plan token ids until matched |

Executive poll output is JSON-serializable (`ExecutivePoll.to_dict()`). A host process can run the poller on turn context, map token strings to GGUF vocab ids, and apply bias before sampling.

## Minimal reference loop

See [`scripts/demo_executive_bias_numpy.py`](../scripts/demo_executive_bias_numpy.py): NumPy logits, synthetic influence, greedy argmax—shows delta vs unbiased decode without pulling in llama.cpp. Swap the logits source for your GGUF binding’s pre-softmax buffer.

## Evaluation angle

For hiring-facing demos: pair the same prompt with bias off vs on, hold temperature fixed, compare plan-token coverage rate on a short checklist utterance. Release gates in `brain_regions/evaluation/articulator_release_gate.py` define the in-repo metric; GGUF is an external corroboration path.
