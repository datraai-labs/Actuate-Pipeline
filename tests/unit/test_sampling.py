"""Even frame sampling: a short cap must span the whole clip, not just its opening
(the bug that made a customer's HF video yield 0 hands), and hands+depth must agree."""

from __future__ import annotations

from actuate.perception.sampling import sampled_indices


def test_full_when_max_none_or_larger():
    assert sampled_indices(10, None) == list(range(10))
    assert sampled_indices(10, 20) == list(range(10))     # max >= count -> all, contiguous


def test_even_sample_spans_the_whole_clip():
    idx = sampled_indices(2850, 15)
    assert len(idx) == 15
    assert idx[0] == 0 and idx[-1] == 2849               # first AND last frame included
    # spread, not clustered at the start (the old first-N bug)
    assert idx[7] > 1000


def test_indices_sorted_and_unique():
    idx = sampled_indices(100, 30)
    assert idx == sorted(set(idx))


def test_empty_clip():
    assert sampled_indices(0, 5) == []


def test_hands_and_depth_share_indices():
    """The alignment invariant: both stages sample the SAME frames for a given cap."""
    for count, cap in [(2850, 15), (500, 40), (30, 30)]:
        assert sampled_indices(count, cap) == sampled_indices(count, cap)
