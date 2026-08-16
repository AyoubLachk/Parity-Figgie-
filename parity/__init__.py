"""Parity: a Figgie exchange, Bayesian agents, and a tournament harness."""

from parity.dealer import Deal, deal
from parity.rng import GameStreams, game_streams, spawn
from parity.settlement import check_conservation, settle
from parity.types import (
    ALL_DECKS,
    DECK_SIZE,
    GOAL_CARD_VALUE,
    POT,
    Colour,
    DeckComposition,
    Suit,
    SuitCounts,
    uniform_prior,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_DECKS",
    "DECK_SIZE",
    "GOAL_CARD_VALUE",
    "POT",
    "Colour",
    "Deal",
    "DeckComposition",
    "GameStreams",
    "Suit",
    "SuitCounts",
    "check_conservation",
    "deal",
    "game_streams",
    "settle",
    "spawn",
    "uniform_prior",
]