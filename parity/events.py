"""The event log's alphabet.

The rule that shapes this module: **game state is a fold over the event log**.
If you cannot reconstruct exactly who holds which cards, who has how much cash,
and what is resting in each book by replaying these events from the start, then
the alphabet is incomplete -- and you find that out on tournament day, when the
replay viewer disagrees with the results table.

Four consumers pull in different directions:

  * replay          needs completeness  -- every state change is an event
  * opponent models need attribution    -- who did what, and was it voluntary
  * the tournament  needs cheapness     -- millions of these get constructed
  * the viewer      needs serialisability

Ground truth (the deck, everyone's hands) lives in the log too. `redact()` is
the single function that decides what a given seat is allowed to see, so
"can an agent cheat?" is one testable function rather than a property you hope
holds everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from parity.orders import (
    CancelReason,
    Order,
    OrderId,
    Price,
    Quantity,
    RejectReason,
    Seat,
    Side,
    Trade,
)
from parity.types import DeckComposition, Suit, SuitCounts

# ------------------------------------------------------------ setup events


@dataclass(frozen=True, slots=True)
class GameStarted:
    n_seats: int
    event_budget: int
    starting_cash: int
    seq: int


@dataclass(frozen=True, slots=True)
class DeckRevealed:
    """Ground truth. Redacted from every agent's view until the game ends."""

    composition: DeckComposition
    seq: int


@dataclass(frozen=True, slots=True)
class HandDealt:
    """Ground truth. Redacted from every seat except its owner."""

    seat: Seat
    hand: SuitCounts
    seq: int


# ----------------------------------------------------------- order events


@dataclass(frozen=True, slots=True)
class OrderPosted:
    order: Order
    seq: int


@dataclass(frozen=True, slots=True)
class OrderRejected:
    """A refused order.

    An event rather than an exception: agents *will* submit invalid orders, and
    at game 40,000 you want a countable log entry, not a stack trace.
    """

    seat: Seat
    suit: Suit
    side: Side
    price: Price
    quantity: Quantity
    reason: RejectReason
    seq: int


@dataclass(frozen=True, slots=True)
class OrderCancelled:
    order_id: OrderId
    seat: Seat
    suit: Suit
    reason: CancelReason
    seq: int


@dataclass(frozen=True, slots=True)
class Traded:
    trade: Trade
    seq: int


@dataclass(frozen=True, slots=True)
class Passed:
    """A seat chose to do nothing. "How often does this agent pass" is a real
    diagnostic, and without the event the log cannot tell passing from a seat
    never being asked."""

    seat: Seat
    seq: int


# ------------------------------------------------------------ end of game


@dataclass(frozen=True, slots=True)
class GameEnded:
    reason: str
    seq: int


@dataclass(frozen=True, slots=True)
class Settled:
    final_hands: tuple[SuitCounts, ...]
    payouts: tuple[int, ...]
    seq: int


Event = (
    GameStarted
    | DeckRevealed
    | HandDealt
    | OrderPosted
    | OrderRejected
    | OrderCancelled
    | Traded
    | Passed
    | GameEnded
    | Settled
)

#: Events carrying information no agent may see mid-game.
PRIVATE = (DeckRevealed, HandDealt)


def redact(log: list[Event], seat: Seat) -> list[Event]:
    """What `seat` is allowed to observe.

    Drops the deck composition and every other seat's hand. Everything else --
    orders, rejections, cancels, trades -- is public in Figgie, including who
    traded with whom, which is exactly what an opponent model needs.
    """
    visible: list[Event] = []
    for event in log:
        if isinstance(event, DeckRevealed):
            continue
        if isinstance(event, HandDealt) and event.seat != seat:
            continue
        visible.append(event)
    return visible