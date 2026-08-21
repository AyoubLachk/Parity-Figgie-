"""Rule tests.

These are the tests that stop the economics drifting. Every one of them
corresponds to a rule on figgie.com.
"""

from __future__ import annotations

import pytest

from parity.types import (
    ALL_DECKS,
    DECK_SIZE,
    POT,
    SUIT_SIZES,
    Colour,
    DeckComposition,
    Suit,
    uniform_prior,
)

# ------------------------------------------------------------------- suits


def test_colours_partition_the_suits() -> None:
    black = {s for s in Suit if s.colour is Colour.BLACK}
    red = {s for s in Suit if s.colour is Colour.RED}
    assert black == {Suit.SPADES, Suit.CLUBS}
    assert red == {Suit.HEARTS, Suit.DIAMONDS}


@pytest.mark.parametrize("suit", list(Suit))
def test_partner_is_an_involution_of_the_same_colour(suit: Suit) -> None:
    assert suit.partner != suit
    assert suit.partner.colour is suit.colour
    assert suit.partner.partner is suit


# ------------------------------------------------------- the 12 deck space


def test_there_are_exactly_twelve_decks_and_they_are_distinct() -> None:
    assert len(ALL_DECKS) == 12
    assert len(set(ALL_DECKS)) == 12


@pytest.mark.parametrize("deck", ALL_DECKS, ids=str)
def test_every_deck_is_a_permutation_of_12_10_10_8(deck: DeckComposition) -> None:
    assert sorted(deck.counts) == sorted(SUIT_SIZES)
    assert sum(deck.counts) == DECK_SIZE


@pytest.mark.parametrize("deck", ALL_DECKS, ids=str)
def test_goal_suit_shares_the_common_suits_colour(deck: DeckComposition) -> None:
    assert deck[deck.common_suit] == 12
    assert deck.goal_suit is not deck.common_suit
    assert deck.goal_suit.colour is deck.common_suit.colour


def test_goal_suit_is_not_always_the_eight_card_suit() -> None:
    """Regression test for the most expensive misreading of the rules.

    The goal suit is the partner of the 12-card suit, NOT the 8-card suit. It
    happens to be the 8-card suit in only 4 of the 12 decks. Encoding the wrong
    version silently corrupts every posterior and every payout in the project,
    so it gets a named test rather than an assertion buried in a loop.
    """
    eights = [d for d in ALL_DECKS if d.goal_suit_size == 8]
    tens = [d for d in ALL_DECKS if d.goal_suit_size == 10]

    assert len(eights) == 4, "one per choice of common suit"
    assert len(tens) == 8
    assert len(eights) + len(tens) == len(ALL_DECKS)

    # So a priori the goal suit has 10 cards twice as often as it has 8.
    assert len(tens) / len(ALL_DECKS) == pytest.approx(2 / 3)


@pytest.mark.parametrize("deck", ALL_DECKS, ids=str)
def test_bonus_and_threshold_track_goal_suit_size(deck: DeckComposition) -> None:
    if deck.goal_suit_size == 8:
        assert deck.bonus == 120
        assert deck.majority_threshold == 5
    else:
        assert deck.goal_suit_size == 10
        assert deck.bonus == 100
        assert deck.majority_threshold == 6

    # The pot is fully accounted for: every goal card pays 10, the rest is bonus.
    assert deck.bonus + 10 * deck.goal_suit_size == POT


def test_each_suit_is_the_goal_suit_in_exactly_three_decks() -> None:
    """Uniform prior over decks implies a uniform 1/4 prior over goal suits."""
    for suit in Suit:
        assert sum(1 for d in ALL_DECKS if d.goal_suit is suit) == 3


# -------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "counts",
    [
        (12, 10, 10, 10),  # sums to 42
        (12, 12, 8, 8),  # two common suits
        (10, 10, 10, 10),  # no common suit
        (12, 10, 10),  # wrong length
    ],
)
def test_illegal_compositions_are_rejected(counts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        DeckComposition(counts)  # type: ignore[arg-type]


def test_uniform_prior_is_a_probability_vector() -> None:
    prior = uniform_prior()
    assert len(prior) == len(ALL_DECKS)
    assert sum(prior) == pytest.approx(1.0)
    assert all(p > 0 for p in prior)