"""#44: can a predictor send a layer's CPU-served experts to a GPU before the router asks?  prefetch_sim scores a
prediction per (round, layer): a layer gains only when EVERY expert the CPU would have served is covered, because the
layer waits for its slowest path."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import prefetch_sim as ps  # noqa: E402


def test_a_layer_counts_only_when_every_cpu_expert_is_covered():
    # CPU-served experts {3, 7}; a prediction that holds 3 but not 7 covers one entry of two and not the layer
    r = ps.score({3, 7}, [3, 5])
    assert r == {"layer": False, "experts": 1, "of": 2}
    assert ps.score({3, 7}, [7, 3])["layer"] is True


def test_a_layer_with_no_cpu_expert_is_free_without_a_prediction():
    assert ps.score(set(), [])["layer"] is True


def test_last_round_predicts_the_experts_this_layer_used_most_recently():
    p = ps.LastRound(k=2)
    p.observe(layer=4, experts=[10, 11, 12])
    p.observe(layer=4, experts=[12, 13])
    assert p.predict(layer=4, prev_layer_experts=[]) == [12, 13]
    assert p.predict(layer=5, prev_layer_experts=[]) == []


def test_cross_layer_ranks_by_how_often_the_previous_layers_experts_led_to_each_expert():
    p = ps.CrossLayer(k=1)
    p.train([(0, [1]), (1, [8])])            # expert 1 at layer 0 led to expert 8 at layer 1
    p.train([(0, [1, 2]), (1, [8, 9])])
    p.train([(0, [2]), (1, [9])])
    assert p.predict(layer=1, prev_layer_experts=[1]) == [8]
    assert p.predict(layer=1, prev_layer_experts=[2]) == [9]


def test_cpu_set_excludes_the_gpu_tiers():
    placed = {(4, 1), (4, 2)}
    assert ps.cpu_experts(4, [1, 2, 3, 3, -1], placed) == {3}
