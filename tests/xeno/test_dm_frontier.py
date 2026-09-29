"""#41: would a k-order frontier shrink Dm?  dm_frontier_sim.peak_held() replays one layer's routing: experts produce
their (token, k) rows in a given order, and a token's sum takes row k only after rows 0..k-1 (the fmaf order that
byte identity needs), so a row waits in the frontier until every lower-rank row of its token has been produced."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import dm_frontier_sim as fs  # noqa: E402


def test_rows_wait_for_their_lower_ranks():
    # one token, K = 3, routed to experts 2, 1, 0 in rank order.  In id order, expert 0 gives rank 2 and expert 1 gives
    # rank 1: both wait for rank 0 (expert 2), so two rows are held at the peak.
    assert fs.peak_held([[2, 1, 0]], order=[0, 1, 2]) == 2


def test_an_order_that_follows_the_ranks_holds_nothing():
    assert fs.peak_held([[2, 1, 0]], order=[2, 1, 0]) == 0


def test_rows_of_several_tokens_are_counted_together():
    # token 0: ranks on experts 1, 0; token 1: ranks on experts 0, 1.  Id order: expert 0 gives (t0, k1), which waits,
    # and (t1, k0), which is taken at once; expert 1 then completes both tokens.
    assert fs.peak_held([[1, 0], [0, 1]], order=[0, 1]) == 1


def test_the_rank_order_heuristic_puts_low_mean_rank_experts_first():
    order = fs.order_by_mean_rank([[2, 1, 0], [2, 0, 1]], n_expert=3)
    assert order[0] == 2   # expert 2 is rank 0 for both tokens
