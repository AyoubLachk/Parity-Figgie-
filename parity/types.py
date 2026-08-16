"""Core value types for Figgie.

Everything here is immutable, hashable, and knows nothing about randomness,
trading or agents. That separation is deliberate: this module is the single
place where the *rules* live, so there is exactly one thing to get right.

Rules encoded here are verified against https://www.figgie.com/how-to-play.html
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

# ---------------------------------------------------------------- constants

DECK_SIZE = 40
SUIT_SIZES = (12, 10, 10, 8)  # the multiset every deck must be a permutation of
POT = 200  # four players ante 50 each
GOAL_CARD_VALUE = 10  # each goal-suit card pays this from the pot
LEGAL_PLAYER_COUNTS = (4, 5)  # 40 must divide evenly into hands


# -------------------------------------------------------------------- suits


class Colour(Enum):
    BLACK = "black"
    RED = "red"


class Suit(IntEnum):
    """Suits as an IntEnum so a Suit can index a length-4 array directly.

    `counts[Suit.CLUBS]` reads better than `counts[1]` and costs nothing.
    """

    SPADES = 0
    CLUBS = 1
    HEARTS = 2
    DIAMONDS = 3

    @property
    def colour(self) -> Colour:
        return Colour.BLACK if self in (Suit.SPADES, Suit.CLUBS) else Colour.RED

    @property
    def partner(self) -> Suit:
        """The other suit of the same colour.

        This is the whole goal-suit rule in one line: goal = common.partner.
        """
        return _PARTNER[self]


_PARTNER: dict[Suit, Suit] = {
    Suit.SPADES: Suit.CLUBS,
    Suit.CLUBS: Suit.SPADES,
    Suit.HEARTS: Suit.DIAMONDS,
    Suit.DIAMONDS: Suit.HEARTS,
}

# A hand, or any bag of cards, is fully described by its four suit counts.
# Rank is irrelevant in Figgie, so suit counts are a *sufficient statistic*:
# nothing is lost by throwing rank away, and the state space collapses.
SuitCounts = tuple[int, int, int, int]


# ------------------------------------------------------------- composition


@dataclass(frozen=True, slots=True)
class DeckComposition:
    """One of the 12 legal ways to build a Figgie deck.

    Exactly one suit has 12 cards (the *common* suit), exactly one has 8, and
    the remaining two have 10. The *goal* suit is the suit of the same colour
    as the common suit -- which means the goal suit has 8 cards only 1 time in
    3, and 10 cards the other 2 times in 3.
    """

    counts: SuitCounts

    def __post_init__(self) -> None:
        if len(self.counts) != 4:
            raise ValueError(f"need 4 suit counts, got {len(self.counts)}")
        if sorted(self.counts) != sorted(SUIT_SIZES):
            raise ValueError(
                f"counts {self.counts} is not a permutation of {SUIT_SIZES}"
            )

    def __getitem__(self, suit: Suit) -> int:
        return self.counts[suit]

    @property
    def common_suit(self) -> Suit:
        """The 12-card suit. Never worth anything itself."""
        return Suit(self.counts.index(12))

    @property
    def goal_suit(self) -> Suit:
        """The only suit that pays. Same colour as the common suit."""
        return self.common_suit.partner

    @property
    def goal_suit_size(self) -> int:
        """8 or 10. Determines both the majority threshold and the bonus."""
        return self.counts[self.goal_suit]

    @property
    def majority_threshold(self) -> int:
        """Goal cards needed to guarantee outright majority: 5 of 8, or 6 of 10."""
        return self.goal_suit_size // 2 + 1

    @property
    def bonus(self) -> int:
        """Pot remainder paid to the holder(s) of the most goal cards.

        POT - 10 * goal_suit_size, i.e. 120 when the goal suit has 8 cards and
        100 when it has 10. The smaller goal suit carries the bigger prize.
        """
        return POT - GOAL_CARD_VALUE * self.goal_suit_size

    def __str__(self) -> str:
        body = " ".join(f"{s.name[0]}{self.counts[s]}" for s in Suit)
        return f"<{body} | goal={self.goal_suit.name}({self.goal_suit_size})>"


def _enumerate_decks() -> tuple[DeckComposition, ...]:
    """All 12 legal decks: 4 choices of common suit x 3 choices of 8-card suit."""
    decks: list[DeckComposition] = []
    for common in Suit:
        for eight in Suit:
            if eight == common:
                continue
            counts = [10, 10, 10, 10]
            counts[common] = 12
            counts[eight] = 8
            decks.append(DeckComposition(tuple(counts)))  # type: ignore[arg-type]
    return tuple(decks)


ALL_DECKS: tuple[DeckComposition, ...] = _enumerate_decks()
"""The full hypothesis space. Your posterior is a length-12 vector over this."""


def uniform_prior() -> tuple[float, ...]:
    """Prior over ALL_DECKS before seeing any cards. Uniform by construction."""
    return tuple([1.0 / len(ALL_DECKS)] * len(ALL_DECKS))
