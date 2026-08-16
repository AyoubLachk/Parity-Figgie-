"""Dealing.

Picks one of the 12 legal decks uniformly at random, shuffles the 40 physical
cards, and splits them evenly between players. Consumes exactly one RNG stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parity.rng import Generator
from parity.types import (
    ALL_DECKS,
    DECK_SIZE,
    LEGAL_PLAYER_COUNTS,
    DeckComposition,
    Suit,
    SuitCounts,
)


@dataclass(frozen=True, slots=True)
class Deal:
    """The ground truth for one round. Agents never see `composition`."""

    composition: DeckComposition
    hands: tuple[SuitCounts, ...]

    @property
    def n_players(self) -> int:
        return len(self.hands)

    @property
    def cards_per_player(self) -> int:
        return DECK_SIZE // self.n_players

    def total_by_suit(self) -> SuitCounts:
        """Cards of each suit held across all players. Must equal composition."""
        totals = [0, 0, 0, 0]
        for hand in self.hands:
            for suit in Suit:
                totals[suit] += hand[suit]
        return tuple(totals)  # type: ignore[return-value]

    def goal_cards_held(self) -> tuple[int, ...]:
        goal = self.composition.goal_suit
        return tuple(hand[goal] for hand in self.hands)


def deal(rng: Generator, n_players: int = 4) -> Deal:
    """Deal one round.

    Two independent random choices, in a fixed order so the stream is stable:
      1. which of the 12 deck compositions is in play;
      2. how the 40 physical cards are permuted.
    """
    if n_players not in LEGAL_PLAYER_COUNTS:
        raise ValueError(
            f"n_players must be one of {LEGAL_PLAYER_COUNTS}, got {n_players}"
        )

    composition = ALL_DECKS[int(rng.integers(len(ALL_DECKS)))]

    # Materialise the physical deck as 40 suit labels, then permute it. We do
    # not sample hands directly from the multivariate hypergeometric, even
    # though the distribution is identical -- shuffling a real deck is easier
    # to check by eye and impossible to get subtly wrong.
    deck = np.repeat(np.arange(4), composition.counts)
    rng.shuffle(deck)

    per_player = DECK_SIZE // n_players
    hands: list[SuitCounts] = []
    for i in range(n_players):
        block = deck[i * per_player : (i + 1) * per_player]
        counts = np.bincount(block, minlength=4)
        hands.append(tuple(int(c) for c in counts))  # type: ignore[arg-type]

    return Deal(composition=composition, hands=tuple(hands))