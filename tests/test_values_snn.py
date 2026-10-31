import numpy as np

from brain_regions.agents.values import ValuesSNN


def test_lantern_words_receive_excitation():
    snn = ValuesSNN()
    out = snn.tick({
        "word_candidates": ["empathy", "table"],
        "value_activations": {"empathy": 0.6},
        "output_tokens": [],
    })
    assert out.bias_vector[0] > out.bias_vector[1]
    assert out.spikes.shape == (128,)


def test_poison_pair_inhibited_after_host_spoken():
    snn = ValuesSNN()
    out = snn.tick({
        "word_candidates": ["compassion", "destroy", "table"],
        "value_activations": {},
        "output_tokens": ["we", "destroy"],
    })
    assert out.inhibition_vector[0] > 0
    assert out.inhibition_vector[1] > 0
    assert out.inhibition_vector[2] == 0


def test_values_resist_change_below_challenge_threshold():
    snn = ValuesSNN()
    before = snn.reservoir.W_in.copy()
    assert snn.apply_value_delta("cruelty", 0.5, challenge_strength=0.1) is False
    assert np.array_equal(before, snn.reservoir.W_in)
