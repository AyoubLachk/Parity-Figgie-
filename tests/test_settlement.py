"""Settlement tests.

The property that matters: the pot is conserved. Chips are redistributed, never
created. If this holds for every reachable final allocation, agent PnL is a
zero-sum game and any "profit" you measure later is real.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from parity.dealer import deal
from parity.settlement import (
    check_conservation,
    marginal_goal_card_value,
    settle,
)
from parity.types import ALL_DECKS, POT, DeckComposition, Suit, SuitCounts

seeds = st.integers(min_value=0, max_value=2**32 - 1)


def _random_final_allocation(
    rng: np.random.Generator, composition: DeckComposition, n_players: int
) -> tuple[SuitCounts, ...]:
    """Any allocation trading could plausibly reach: all 40 cards, any split.

    Hands need not be equal sized -- players can finish with no cards at all.
    """
    hands = [[0, 0, 0, 0] for _ in range(n_players)]
    for suit in Suit:
        owners = rng.integers(0, n_players, size=composition[suit])
        for owner in owners:
            hands[int(owner)][suit] += 1
    return tuple((h[0], h[1], h[2], h[3]) for h in hands)


# ------------------------------------------------------- pot conservation


@given(seed=seeds, n_players=st.sampled_from([4, 5]))
def test_settlement_always_pays_out_exactly_the_pot(seed: int, n_players: int) -> None:
    rng = np.random.default_rng(seed)
    composition = ALL_DECKS[int(rng.integers(len(ALL_DECKS)))]
    hands = _random_final_allocation(rng, composition, n_players)

    payouts = settle(composition, hands)

    assert sum(payouts) == POT
    assert all(p >= 0 for p in payouts)
    assert len(payouts) == n_players


@given(seed=seeds)
def test_the_dealt_hands_settle_correctly_before_any_trading(seed: int) -> None:
    d = deal(np.random.default_rng(seed))
    assert sum(settle(d.composition, d.hands)) == POT


@given(seed=seeds, n_players=st.sampled_from([4, 5]))
def test_holding_more_goal_cards_never_pays_less(seed: int, n_players: int) -> None:
    """Monotonicity. Obvious, and precisely the kind of obvious thing that breaks
    when someone edits the tie-splitting branch six weeks from now."""
    rng = np.random.default_rng(seed)
    composition = ALL_DECKS[int(rng.integers(len(ALL_DECKS)))]
    hands = _random_final_allocation(rng, composition, n_players)
    payouts = settle(composition, hands)

    goal = composition.goal_suit
    for i in range(n_players):
        for j in range(n_players):
            if hands[i][goal] > hands[j][goal]:
                assert payouts[i] > payouts[j]


# ----------------------------------------------------------- known answers


def test_monopolist_takes_the_whole_pot() -> None:
    """8 clubs goal suit: 8 * 10 = 80 in card value, plus a 120 bonus = 200."""
    composition = DeckComposition((12, 8, 10, 10))  # spades common, clubs goal
    assert composition.goal_suit is Suit.CLUBS
    assert composition.goal_suit_size == 8

    hands = ((12, 8, 10, 10), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    assert settle(composition, hands) == (POT, 0, 0, 0)


def test_two_way_tie_splits_the_bonus() -> None:
    """8-card goal suit, 4 each: 40 + 60 = 100 apiece."""
    composition = DeckComposition((12, 8, 10, 10))
    hands = ((12, 4, 0, 0), (0, 4, 10, 10), (0, 0, 0, 0), (0, 0, 0, 0))
    assert settle(composition, hands) == (100, 100, 0, 0)


def test_ten_card_goal_suit_pays_a_smaller_bonus() -> None:
    """Clubs common, spades goal with 10 cards: bonus is 100, not 120."""
    composition = DeckComposition((10, 12, 10, 8))
    assert composition.goal_suit is Suit.SPADES
    assert composition.goal_suit_size == 10
    assert composition.bonus == 100

    hands = ((10, 0, 0, 0), (0, 12, 10, 8), (0, 0, 0, 0), (0, 0, 0, 0))
    assert settle(composition, hands) == (200, 0, 0, 0)


def test_four_way_tie_divides_the_bonus_exactly() -> None:
    """8-card goal suit, 2 each: 20 + 30 = 50 apiece, no rounding needed."""
    eight = DeckComposition((12, 8, 10, 10))
    hands = ((12, 2, 0, 0), (0, 2, 10, 0), (0, 2, 0, 10), (0, 2, 0, 0))

    assert settle(eight, hands) == (50, 50, 50, 50)


def test_three_way_tie_leftover_chips_go_to_low_seats() -> None:
    """100 split three ways is 33 each with 1 chip left over.

    The leftover goes to the lowest seat by the house rule documented in
    settle(). The point of the test is not the rule -- any rule would do -- but
    that the rule is *fixed*, so the pot still balances to the chip and results
    stay reproducible.
    """
    ten = DeckComposition((10, 12, 10, 8))  # spades goal, 10 cards, bonus 100
    hands = ((3, 4, 0, 0), (3, 4, 0, 0), (3, 4, 10, 8), (1, 0, 0, 0))

    payouts = settle(ten, hands)

    assert payouts == (30 + 34, 30 + 33, 30 + 33, 10)
    assert sum(payouts) == POT


# ----------------------------------------------------------- conservation


def test_conservation_check_rejects_a_missing_card() -> None:
    composition = DeckComposition((12, 8, 10, 10))
    hands = ((12, 8, 10, 9), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="conservation"):
        check_conservation(composition, hands)


def test_conservation_check_rejects_a_duplicated_card() -> None:
    composition = DeckComposition((12, 8, 10, 10))
    hands = ((12, 8, 10, 10), (0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="conservation"):
        check_conservation(composition, hands)


def test_settle_refuses_to_price_an_inconsistent_book() -> None:
    composition = DeckComposition((12, 8, 10, 10))
    hands = ((1, 1, 1, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    with pytest.raises(ValueError):
        settle(composition, hands)


# ------------------------------------------------- the kink, for later use


def test_marginal_card_value_has_a_kink_at_the_lead() -> None:
    """Why the 12-deck posterior matters, in one test.

    The nth goal card is worth 10 chips, except the one that takes you into the
    lead, which is worth 10 + bonus. Where that card sits depends on whether the
    goal suit has 8 cards or 10 -- something you are uncertain about while you
    are still trading.
    """
    eight = DeckComposition((12, 8, 10, 10))  # bonus 120

    # Rival holds 3. Going 2 -> 3 only ties; 3 -> 4 takes the lead outright.
    assert marginal_goal_card_value(eight, my_goal_cards=1, best_rival=3) == 10
    assert marginal_goal_card_value(eight, my_goal_cards=2, best_rival=3) == 10 + 60
    assert marginal_goal_card_value(eight, my_goal_cards=3, best_rival=3) == 10 + 60

    # A card bought while already clear ahead is worth face value only.
    assert marginal_goal_card_value(eight, my_goal_cards=5, best_rival=3) == 10