"""#41: make_expert_order.py turns prefill route traces into a static per-layer expert order (mean router rank first)
for STRATA_EXPERT_ORDER.  Every layer must be a full permutation, or the engine refuses the file."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import make_expert_order as mo  # noqa: E402


def test_experts_run_by_mean_rank_and_unseen_ones_come_last_in_id_order():
    # layer 0: expert 2 always rank 0, expert 0 rank 1, expert 3 rank 2 then 0 (mean 1); expert 1 never routed
    routes = {0: [[2, 0, 3], [2, 3, 0]]}
    order = mo.order_from_routes(routes, n_layers=1, n_expert=4)
    assert order[0] == [2, 0, 3, 1] or order[0] == [2, 3, 0, 1]   # 0 and 3 tie at mean rank 1.5 / 1.5
    assert sorted(order[0]) == [0, 1, 2, 3]


def test_ties_break_by_id_so_the_file_is_deterministic():
    routes = {0: [[1, 0], [0, 1]]}   # both experts at mean rank 0.5
    assert mo.order_from_routes(routes, n_layers=1, n_expert=2)[0] == [0, 1]


def test_the_file_round_trips(tmp_path):
    order = [[1, 0, 2], [2, 1, 0]]
    p = tmp_path / "o.bin"
    mo.write_order(str(p), order)
    assert mo.read_order(str(p)) == order
