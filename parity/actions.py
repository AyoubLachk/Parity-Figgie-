"""What crosses the wall between agents and the exchange.

An agent never touches a book. It receives an immutable `Observation` and
returns an `Action`. That boundary is both a correctness guarantee -- nothing
can mutate the book from outside -- and an anti-cheating one, since the
Observation is built by redaction and simply does not contain the deck.

`Observation` is deliberately small. It is constructed once per turn, millions
of times over a tournament, so it holds only what an agent needs *right now*.
Anything historical the agent wants, it accumulates itself in `on_event` --
which is also how real trading systems work.
"""

from __future__ import annotations

from dataclasses import dataclass

from parity.orders import Order, OrderId, Price, Seat, Side
from parity.types import Suit, SuitCounts


@dataclass(frozen=True, slots=True)
class Quote:
    """Top of book for one suit."""

    bid: Order | None
    ask: Order | None

    @property
    def bid_price(self) -> int | None:
        return None if self.bid is None else int(self.bid.price)

    @property
    def ask_price(self) -> int | None:
        return None if self.ask is None else int(self.ask.price)

    @property
    def spread(self) -> int | None:
        if self.bid is None or self.ask is None:
            return None
        return int(self.ask.price) - int(self.bid.price)

    def __str__(self) -> str:
        b = self.bid_price if self.bid_price is not None else "--"
        a = self.ask_price if self.ask_price is not None else "--"
        return f"{b}/{a}"


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything a seat may see at the moment it is asked to act."""

    seat: Seat
    hand: SuitCounts
    cash: int
    quotes: dict[Suit, Quote]
    my_orders: tuple[Order, ...]
    events_remaining: int

    def holdings(self, suit: Suit) -> int:
        return self.hand[suit]

    def most_held(self) -> Suit:
        """The suit this seat holds most of. Ties break by suit order."""
        return max(Suit, key=lambda s: self.hand[s])

    def least_held(self) -> Suit:
        return min(Suit, key=lambda s: self.hand[s])


# ------------------------------------------------------------------ actions


@dataclass(frozen=True, slots=True)
class PostOrder:
    """Post a new order.

    No `order_id` field: the exchange assigns ids. If agents chose their own,
    one could forge an id and cancel a rival's quote.
    """

    suit: Suit
    side: Side
    price: Price

    def __str__(self) -> str:
        verb = "buy" if self.side is Side.BUY else "sell"
        return f"{verb} {self.suit.name.lower()} @{self.price}"


@dataclass(frozen=True, slots=True)
class CancelOrder:
    order_id: OrderId

    def __str__(self) -> str:
        return f"cancel #{self.order_id}"


@dataclass(frozen=True, slots=True)
class Pass:
    """Do nothing this turn.

    A real action rather than `None`, so the log can tell "chose to pass" from
    "was never asked". How often an agent passes is a genuine diagnostic.
    """

    def __str__(self) -> str:
        return "pass"


Action = PostOrder | CancelOrder | Pass