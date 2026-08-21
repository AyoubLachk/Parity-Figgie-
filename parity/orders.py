"""The exchange's vocabulary.

Records only -- no behaviour lives here. The order book and the exchange own all
the logic; these are the nouns they pass around.

Decisions made here, and the reasons:

1. Prices are int. Never float. Figgie prices are whole chips, and float
   equality is unreliable -- `price_a == price_b` failing by 1e-16 silently
   breaks price-time priority. Every real matching engine uses integers.

2. Seats are not agents. The engine deals in Seat (0..n-1); the tournament maps
   agents onto seats and rotates them. If the engine knew agent *names*, seat
   rotation would be a special case instead of a parameter.

3. Everything is frozen. If Order were mutable and the book mutated it during
   matching, the event log would retroactively become a lie. Frozen types make
   that class of bug impossible.

4. Liveness is NOT on the Order. Remaining quantity and cancelled-ness are the
   book's private bookkeeping. The Order you put in the log never changes.

5. The exchange assigns order_id, not the agent. Otherwise an agent could forge
   an id and cancel someone else's order.

6. seq is a monotonic counter, not a timestamp. Wall clocks are
   non-deterministic and would destroy replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

from parity.types import Suit

OrderId = NewType("OrderId", int)
Seat = NewType("Seat", int)
Price = NewType("Price", int)
Quantity = NewType("Quantity", int)


class Side(Enum):
    """Order intent.

    Use BUY/SELL for intent and bid/ask for book queries (`best_bid`). Mixing
    the vocabularies produces the eternal question "is Side.BID an order or a
    price level?"
    """

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class RejectReason(Enum):
    """Why the exchange refused an order.

    An enum rather than a string so you can count rejections by reason in the
    tournament output and spot a broken agent from the summary table.
    """

    INSUFFICIENT_CASH = "insufficient_cash"
    NO_INVENTORY = "no_inventory"
    SELF_TRADE = "self_trade"
    UNKNOWN_ORDER = "unknown_order"
    NOT_YOUR_ORDER = "not_your_order"
    INVALID_PRICE = "invalid_price"
    INVALID_QUANTITY = "invalid_quantity"
    GAME_OVER = "game_over"


class CancelReason(Enum):
    """Why an order left the book.

    This distinction earns its keep in the opponent model: a voluntary cancel
    carries information about a player's beliefs, an automatic post-trade sweep
    carries none. Collapse them into one event and the model learns from noise.
    """

    EXPLICIT = "explicit"
    TRADE_SWEEP = "trade_sweep"


@dataclass(frozen=True, slots=True)
class Order:
    """An order as posted. Never mutated."""

    order_id: OrderId
    seat: Seat
    suit: Suit
    side: Side
    price: Price
    quantity: Quantity
    seq: int

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")

    def __str__(self) -> str:
        verb = "B" if self.side is Side.BUY else "S"
        return (
            f"#{self.order_id} seat{self.seat} {verb} "
            f"{self.quantity}x{self.suit.name[0]} @{self.price}"
        )


@dataclass(frozen=True, slots=True)
class Trade:
    """One execution.

    `aggressor` is the side that crossed the spread. This matters more than it
    looks: a trade where you were lifted means something very different from a
    trade where you lifted, and adverse-selection updates depend on knowing
    which happened.

    Price is always the RESTING order's price -- the passive side set the terms.
    """

    suit: Suit
    price: Price
    quantity: Quantity
    buyer: Seat
    seller: Seat
    aggressor: Side
    resting_order_id: OrderId
    incoming_order_id: OrderId
    seq: int

    def __str__(self) -> str:
        return (
            f"TRADE {self.quantity}x{self.suit.name[0]} @{self.price} "
            f"seat{self.seller}->seat{self.buyer} ({self.aggressor.value} agg)"
        )