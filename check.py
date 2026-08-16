"""Throwaway sanity check. Run with: python check.py

Not part of the project -- delete it once your real tests exist. It just walks
through every file you typed and reports which one has a problem.
"""

from collections.abc import Callable
from typing import Any

PASS, FAIL = "  OK  ", " FAIL "
failures: list[str] = []


def check(label: str, fn: Callable[[], Any]) -> None:
    """Run one check, print the result, never crash."""
    try:
        result = fn()
    except Exception as exc:
        failures.append(label)
        print(f"[{FAIL}] {label}\n         {type(exc).__name__}: {exc}")
    else:
        print(f"[{PASS}] {label}" + (f"  ->  {result}" if result is not None else ""))


# --------------------------------------------------------------- imports

print("\n--- imports ---")
check("import parity", lambda: __import__("parity"))

from parity.dealer import deal
from parity.rng import game_streams, spawn
from parity.settlement import marginal_goal_card_value, settle
from parity.types import (
    ALL_DECKS,
    DECK_SIZE,
    POT,
    Colour,
    DeckComposition,
    Suit,
    uniform_prior,
)

# ----------------------------------------------------------------- types

print("\n--- types.py ---")
check("DECK_SIZE is 40", lambda: DECK_SIZE)
check("POT is 200", lambda: POT)
check("4 suits exist", lambda: [s.name for s in Suit])
check("spades is black", lambda: Suit.SPADES.colour is Colour.BLACK)
check("hearts is red", lambda: Suit.HEARTS.colour is Colour.RED)
check("clubs' partner is spades", lambda: Suit.CLUBS.partner.name)
check(
    "partner applied twice returns you home",
    lambda: all(s.partner.partner is s for s in Suit),
)
check("there are 12 legal decks", lambda: len(ALL_DECKS))
check("all 12 are distinct", lambda: len(set(ALL_DECKS)) == 12)
check("every deck sums to 40", lambda: all(sum(d.counts) == 40 for d in ALL_DECKS))
check(
    "exactly 4 decks have an 8-card goal suit",
    lambda: sum(1 for d in ALL_DECKS if d.goal_suit_size == 8),
)
check(
    "8-card goal suit pays a 120 bonus",
    lambda: all(d.bonus == 120 for d in ALL_DECKS if d.goal_suit_size == 8),
)
check(
    "10-card goal suit pays a 100 bonus",
    lambda: all(d.bonus == 100 for d in ALL_DECKS if d.goal_suit_size == 10),
)
check(
    "majority threshold is 5 of 8 or 6 of 10",
    lambda: {(d.goal_suit_size, d.majority_threshold) for d in ALL_DECKS},
)
check("__str__ is readable", lambda: str(ALL_DECKS[0]))
check("uniform_prior sums to 1", lambda: round(sum(uniform_prior()), 10) == 1.0)


def _reject_bad_deck() -> str:
    try:
        DeckComposition((10, 10, 10, 10))
    except ValueError:
        return "raised ValueError as it should"
    raise AssertionError("an illegal deck was accepted!")


check("illegal decks are rejected", _reject_bad_deck)

# ------------------------------------------------------------------- rng

print("\n--- rng.py ---")
check("spawn gives 4 generators", lambda: len(spawn(2026, 4)))
check(
    "spawned streams differ",
    lambda: len({tuple(g.integers(0, 10**9, 5).tolist()) for g in spawn(2026, 4)}) == 4,
)
check("game_streams has 4 agent streams", lambda: len(game_streams(2026, 0, 4).agents))

# ---------------------------------------------------------------- dealer

print("\n--- dealer.py ---")
d = deal(game_streams(2026, 0, 4).deal)
check("a deal happened", lambda: str(d.composition))
check("4 hands were dealt", lambda: len(d.hands))
check("each hand has 10 cards", lambda: all(sum(h) == 10 for h in d.hands))
check("CARD CONSERVATION", lambda: d.total_by_suit() == d.composition.counts)
check("same seed gives same deal", lambda: deal(game_streams(2026, 0, 4).deal) == d)
check("different seeds differ", lambda: deal(game_streams(2026, 1, 4).deal) != d)

# ------------------------------------------------------------ settlement

print("\n--- settlement.py ---")
payouts = settle(d.composition, d.hands)
check("settlement ran", lambda: payouts)
check("POT CONSERVATION (sums to 200)", lambda: sum(payouts) == POT)
check("nobody gets a negative payout", lambda: all(p >= 0 for p in payouts))
check(
    "a monopolist takes the whole pot",
    lambda: settle(
        DeckComposition((12, 8, 10, 10)),
        ((12, 8, 10, 10), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
    )
    == (200, 0, 0, 0),
)
check(
    "the marginal-value kink exists",
    lambda: (
        marginal_goal_card_value(DeckComposition((12, 8, 10, 10)), 1, 3),
        marginal_goal_card_value(DeckComposition((12, 8, 10, 10)), 2, 3),
    ),
)

# ---------------------------------------------------------- what you got

print("\n--- your deal, seed 2026 ---")
c = d.composition
print(f"deck        : {c}")
print(f"common suit : {c.common_suit.name} (12 cards, worth nothing)")
print(f"goal suit   : {c.goal_suit.name} ({c.goal_suit_size} cards)")
print(f"bonus       : {c.bonus} chips, majority at {c.majority_threshold}")
print("             S   C   H   D    goal cards   payout")
for i, h in enumerate(d.hands):
    print(f"  seat {i}   {h[0]:>3} {h[1]:>3} {h[2]:>3} {h[3]:>3}    {h[c.goal_suit]:>8}   {payouts[i]:>6}")
print(f"                                  total: {sum(payouts)}")

print()
if failures:
    print(f"*** {len(failures)} CHECK(S) FAILED ***")
    for f in failures:
        print(f"  - {f}")
else:
    print("*** ALL CHECKS PASSED -- your four files are correct ***")