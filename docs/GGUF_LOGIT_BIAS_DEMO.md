# Applying the same nudge on a local GGUF model

Track C in PyTorch does:

```text
final_logits = lm_logits + α · executive_bias
```

Many local GGUF runtimes (llama.cpp and wrappers) accept a sparse bias on vocabulary scores before sampling. You can reuse the executive poll’s JSON, map word strings to that model’s token ids, and apply the same idea without training a LoRA.

## Mapping

| Track C field | Typical GGUF hook |
|---------------|-------------------|
| `concept_bias[token_id]` | Per-id logit bias |
| `alpha` | Scalar on the bias vector |
| Plan slot coverage | Extra mass on plan word ids until they appear |

`ExecutivePoll.to_dict()` is JSON. A host maps strings → vocab ids and calls the runtime’s bias API each step.

## Reference without GGUF

[`scripts/demo_executive_bias_numpy.py`](../scripts/demo_executive_bias_numpy.py) uses NumPy logits, a synthetic plan, and greedy argmax. Swap in your decoder’s pre-softmax buffer to try the same comparison.

## Measuring it

Hold temperature fixed. Same prompt, bias off vs on. Count how often plan words appear. In-repo gates: `brain_regions/evaluation/articulator_release_gate.py`.
