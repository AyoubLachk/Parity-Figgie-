"""Settlement: turning final hands into chips.

This is the function every agent is ultimately optimising against, so it is the
single most expensive thing in the repo to get wrong. It is also 20 lines, which
is exactly why it deserves its own module and its own property tests.
"""

from __future__ import annotations

from parity.types import (
    GOAL_CARD_VALUE,
    POT,
    DeckComposition,
    Suit,
    SuitCounts,
)


def check_conservation(
    composition: DeckComposition, final_hands: tuple[SuitCounts, ...]
) -> None:
    """Raise unless the final hands still account for exactly the dealt deck.

    Trading moves cards between players; it never creates or destroys them.
    Calling this at settlement turns a whole class of matching-engine bugs into
    a loud failure instead of a slightly wrong PnL number.
    """
    for suit in Suit:
        held = sum(hand[suit] for hand in final_hands)
        if held != composition[suit]:
            raise ValueError(
                f"card conservation violated for {suit.name}: "
                f"{held} held vs {composition[suit]} dealt"
            )


def settle(
    composition: DeckComposition, final_hands: tuple[SuitCounts, ...]
) -> tuple[int, ...]:
    """Chips paid out of the pot to each player, in seat order.

    Two components:
      * every goal-suit card pays GOAL_CARD_VALUE from the pot;
      * the remainder (POT - 10 * goal_suit_size, so 120 or 100) goes to
        whoever holds the most goal cards, split if tied.

    The returned tuple always sums to exactly POT. This is money moving out of a
    fixed pot, not money being created -- a player's net result for the round is
    `settle(...)[i] - ante`, computed by the caller.

    House rule: when a tie split is not a whole number of chips, the leftover
    chips go to the lowest seat indices. The official app splits "evenly" and
    does not specify rounding; making the choice explicit keeps settlement exact
    in integer arithmetic, which keeps the conservation test exact.
    """
    check_conservation(composition, final_hands)

    goal = composition.goal_suit
    held = [hand[goal] for hand in final_hands]

    payouts = [GOAL_CARD_VALUE * c for c in held]

    best = max(held)
    winners = [i for i, c in enumerate(held) if c == best]

    share, leftover = divmod(composition.bonus, len(winners))
    for rank, seat in enumerate(winners):
        payouts[seat] += share + (1 if rank < leftover else 0)

    return tuple(payouts)


def marginal_goal_card_value(
    composition: DeckComposition, my_goal_cards: int, best_rival: int
) -> int:
    """Chips gained by acquiring one more goal card, given rivals' holdings.

    Not used yet, but it is the reason the 12-deck posterior matters: this is
    *not* a flat 10. It is 10 almost everywhere and 10 + bonus at the card that
    takes you from tied-or-behind to outright ahead. That kink is the whole
    game, and its location depends on the goal suit having 8 or 10 cards.
    """
    before = settle_two_player_stub(composition, my_goal_cards, best_rival)
    after = settle_two_player_stub(composition, my_goal_cards + 1, best_rival)
    return after - before


def settle_two_player_stub(
    composition: DeckComposition, mine: int, best_rival: int
) -> int:
    """My payout if I hold `mine` goal cards and the best rival holds `best_rival`."""
    payout = GOAL_CARD_VALUE * mine
    if mine > best_rival:
        payout += composition.bonus
    elif mine == best_rival:
        payout += composition.bonus // 2
    return payout
