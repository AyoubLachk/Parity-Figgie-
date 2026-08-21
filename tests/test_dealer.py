"""Dealer tests.

The headline one is card conservation. The rest exist because a stochastic
simulation you cannot replay is a simulation you cannot debug.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from parity.dealer import deal
from parity.rng import game_streams, spawn
from parity.types import ALL_DECKS, DECK_SIZE, LEGAL_PLAYER_COUNTS, Suit

seeds = st.integers(min_value=0, max_value=2**32 - 1)
player_counts = st.sampled_from(LEGAL_PLAYER_COUNTS)


# ------------------------------------------------------- card conservation


@given(seed=seeds, n_players=player_counts)
def test_card_conservation(seed: int, n_players: int) -> None:
    """Every card dealt is held by exactly one player. No more, no fewer.

    This is the invariant the whole simulation rests on: if it can break here,
    at the simplest point in the system, there is no hope of trusting it after
    ten thousand trades have moved cards around.
    """
    d = deal(np.random.default_rng(seed), n_players)

    assert d.total_by_suit() == d.composition.counts
    assert sum(sum(hand) for hand in d.hands) == DECK_SIZE


@given(seed=seeds, n_players=player_counts)
def test_hands_are_equal_sized_and_non_negative(seed: int, n_players: int) -> None:
    d = deal(np.random.default_rng(seed), n_players)

    assert len(d.hands) == n_players
    for hand in d.hands:
        assert len(hand) == 4
        assert all(c >= 0 for c in hand)
        assert sum(hand) == DECK_SIZE // n_players


@given(seed=seeds)
def test_dealt_composition_is_always_legal(seed: int) -> None:
    d = deal(np.random.default_rng(seed))
    assert d.composition in ALL_DECKS


@given(seed=seeds)
def test_no_hand_can_contain_more_of_a_suit_than_exists(seed: int) -> None:
    d = deal(np.random.default_rng(seed))
    for hand in d.hands:
        for suit in Suit:
            assert hand[suit] <= d.composition[suit]


# ------------------------------------------------------------ determinism


@given(seed=seeds, n_players=player_counts)
def test_same_seed_gives_the_same_deal(seed: int, n_players: int) -> None:
    a = deal(np.random.default_rng(seed), n_players)
    b = deal(np.random.default_rng(seed), n_players)
    assert a == b


@given(master=seeds, game=st.integers(0, 10_000))
def test_game_streams_are_reproducible_in_isolation(master: int, game: int) -> None:
    """Game 7000 deals the same cards whether or not games 0..6999 ever ran.

    This is what lets the tournament harness shard across processes and still
    compare agents on identical deals.
    """
    a = deal(game_streams(master, game, n_agents=4).deal)
    b = deal(game_streams(master, game, n_agents=4).deal)
    assert a == b


def test_consecutive_games_differ() -> None:
    deals = [deal(game_streams(2026, g, n_agents=4).deal) for g in range(200)]
    assert len(set(deals)) > 150, "streams look correlated across game index"


def test_agent_streams_are_independent_of_each_other() -> None:
    streams = game_streams(2026, 0, n_agents=4)
    draws = [tuple(g.integers(0, 10**6, size=20).tolist()) for g in streams.agents]
    assert len(set(draws)) == 4


def test_agent_draws_do_not_disturb_the_deal() -> None:
    """The CRN guarantee: burning agent randomness leaves the deal untouched."""
    baseline = deal(game_streams(2026, 42, n_agents=4).deal)

    streams = game_streams(2026, 42, n_agents=4)
    for g in streams.agents:
        g.integers(0, 10**9, size=1000)  # a chattier agent
    assert deal(streams.deal) == baseline


def test_spawned_streams_are_distinct() -> None:
    gens = spawn(master_seed=7, n=8)
    draws = {tuple(g.integers(0, 10**9, size=10).tolist()) for g in gens}
    assert len(draws) == 8


# ------------------------------------------------------------ distribution


@settings(deadline=None)
@given(st.just(None))
def test_deck_compositions_are_uniform(_: None) -> None:
    """Chi-square goodness of fit against the uniform prior over 12 decks.

    A dealer that quietly favours some decks would bias every posterior you
    ever validate against it, and nothing else in the test suite would notice.
    """
    n = 24_000
    rng = np.random.default_rng(20260814)
    observed = np.zeros(len(ALL_DECKS), dtype=int)
    index = {d: i for i, d in enumerate(ALL_DECKS)}
    for _i in range(n):
        observed[index[deal(rng).composition]] += 1

    expected = n / len(ALL_DECKS)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    # 11 degrees of freedom, 99.9th percentile is 31.26.
    assert chi2 < 31.26, f"non-uniform deck sampling, chi2={chi2:.2f}"


@settings(deadline=None, max_examples=1)
@given(st.just(None))
def test_hand_marginals_match_the_hypergeometric_mean(_: None) -> None:
    """Sanity check on the shuffle: E[suit s in a 10-card hand] = 10 * n_s / 40.

    Cheap now, and it is the same expectation your hypergeometric likelihood
    will be built on, so a mismatch here is a warning about the inference step.
    """
    rng = np.random.default_rng(11)
    target = ALL_DECKS[0]
    totals = np.zeros(4)
    games = 0
    for _i in range(40_000):
        d = deal(rng)
        if d.composition != target:
            continue
        totals += np.array(d.hands[0])
        games += 1

    assert games > 500
    expected = np.array(target.counts) * 10 / 40
    assert np.allclose(totals / games, expected, atol=0.06)


# ---------------------------------------------------------------- guards


@pytest.mark.parametrize("n_players", [0, 1, 2, 3, 6, 7])
def test_illegal_player_counts_are_rejected(n_players: int) -> None:
    with pytest.raises(ValueError):
        deal(np.random.default_rng(0), n_players)