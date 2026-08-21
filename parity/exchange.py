"""The exchange: four books, accounts, validation, and the Figgie sweep.

The order book matches orders for one suit and knows nothing else -- not who
owns what, not whether anyone can afford anything, not that three other suits
exist. This module owns all of that.

Four responsibilities:

  1. ACCOUNTS      -- cards and cash per seat, from the deal to the final hands
  2. VALIDATION    -- reject before matching, never after
  3. TRANSFER      -- the only place inventory changes
  4. THE SWEEP     -- Figgie wipes every quote in every suit after any trade

Deliberately absent: any notion of a goal suit. This is a generic four-
instrument venue that happens to be running Figgie. If the word "goal" appears
here, strategy has leaked into infrastructure -- settlement.py owns that.

RESERVATION
-----------
If you hold one club and post two sell orders, both could fill and you would
have sold a card you never had. So resting orders reserve what they promise:
sells reserve cards, buys reserve cash. Reservations are *derived* from the
live books rather than tracked in a parallel dict -- one source of truth, and
nothing to get out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from parity.book import OrderBook
from parity.dealer import Deal
from parity.events import (
    DeckRevealed,
    Event,
    GameStarted,
    HandDealt,
    OrderCancelled,
    OrderPosted,
    OrderRejected,
    Passed,
    Traded,
)
from parity.orders import (
    CancelReason,
    Order,
    OrderId,
    Price,
    Quantity,
    RejectReason,
    Seat,
    Side,
)
from parity.types import Suit, SuitCounts

DEFAULT_STARTING_CASH = 350

#: Figgie trades one card at a time; a module-level singleton keeps the
#: default out of the signature, which linters rightly dislike.
ONE = Quantity(1)


@dataclass(slots=True)
class Exchange:
    """A four-instrument venue running one round of Figgie."""

    deal: Deal
    cancel_all_on_trade: bool = True
    starting_cash: int = DEFAULT_STARTING_CASH

    _hands: list[list[int]] = field(default_factory=list)
    _cash: list[int] = field(default_factory=list)
    _books: dict[Suit, OrderBook] = field(default_factory=dict)
    _next_order_id: int = 0
    _seq: int = 0
    log: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._hands = [list(hand) for hand in self.deal.hands]
        self._cash = [self.starting_cash] * self.n_seats
        self._books = {suit: OrderBook(suit=suit) for suit in Suit}

        self._emit(
            GameStarted(
                n_seats=self.n_seats,
                event_budget=0,
                starting_cash=self.starting_cash,
                seq=self._tick(),
            )
        )
        self._emit(DeckRevealed(composition=self.deal.composition, seq=self._tick()))
        for seat, hand in enumerate(self.deal.hands):
            self._emit(HandDealt(seat=Seat(seat), hand=hand, seq=self._tick()))

    # ------------------------------------------------------------ plumbing

    @property
    def n_seats(self) -> int:
        return len(self.deal.hands)

    def _tick(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, event: Event) -> None:
        self.log.append(event)

    # --------------------------------------------------------------- reads

    def book(self, suit: Suit) -> OrderBook:
        return self._books[suit]

    def hands(self) -> tuple[SuitCounts, ...]:
        """Current holdings. Feed these to settle() when the round ends."""
        return tuple((h[0], h[1], h[2], h[3]) for h in self._hands)

    def cash(self) -> tuple[int, ...]:
        return tuple(self._cash)

    def reserved_cards(self, seat: Seat, suit: Suit) -> int:
        """Cards promised away by this seat's live sell orders in `suit`."""
        book = self._books[suit]
        return sum(
            book.remaining(o.order_id)
            for o in book.live_orders()
            if o.seat == seat and o.side is Side.SELL
        )

    def reserved_cash(self, seat: Seat) -> int:
        """Cash committed by this seat's live buy orders, across all suits."""
        return sum(
            o.price * book.remaining(o.order_id)
            for book in self._books.values()
            for o in book.live_orders()
            if o.seat == seat and o.side is Side.BUY
        )

    def available_cards(self, seat: Seat, suit: Suit) -> int:
        return self._hands[seat][suit] - self.reserved_cards(seat, suit)

    def available_cash(self, seat: Seat) -> int:
        return self._cash[seat] - self.reserved_cash(seat)

    # -------------------------------------------------------------- writes

    def submit(
        self,
        seat: Seat,
        suit: Suit,
        side: Side,
        price: Price,
        quantity: Quantity = ONE,
    ) -> list[Event]:
        """Validate, match, transfer, sweep. Returns the events produced.

        The exchange -- not the caller -- assigns the order id and sequence
        number. Otherwise an agent could forge an id and cancel a rival's quote.
        """
        reason = self._validate(seat, suit, side, price, quantity)
        if reason is not None:
            event = OrderRejected(
                seat=seat,
                suit=suit,
                side=side,
                price=price,
                quantity=quantity,
                reason=reason,
                seq=self._tick(),
            )
            self._emit(event)
            return [event]

        self._next_order_id += 1
        order = Order(
            order_id=OrderId(self._next_order_id),
            seat=seat,
            suit=suit,
            side=side,
            price=price,
            quantity=quantity,
            seq=self._tick(),
        )

        produced: list[Event] = [OrderPosted(order=order, seq=order.seq)]
        trades = self._books[suit].add(order)

        for trade in trades:
            self._transfer(
                trade.buyer,
                trade.seller,
                suit,
                int(trade.quantity),
                int(trade.price),
            )
            produced.append(Traded(trade=trade, seq=self._tick()))

        for produced_event in produced:
            self._emit(produced_event)

        if trades and self.cancel_all_on_trade:
            produced.extend(self._sweep())

        return produced

    def cancel(self, seat: Seat, order_id: OrderId) -> list[Event]:
        """Cancel one of your own resting orders."""
        for suit, book in self._books.items():
            if book.remaining(order_id) > 0:
                if not book.cancel(order_id, seat=seat):
                    return [self._reject_cancel(seat, RejectReason.NOT_YOUR_ORDER)]
                cancelled = OrderCancelled(
                    order_id=order_id,
                    seat=seat,
                    suit=suit,
                    reason=CancelReason.EXPLICIT,
                    seq=self._tick(),
                )
                self._emit(cancelled)
                return [cancelled]
        return [self._reject_cancel(seat, RejectReason.UNKNOWN_ORDER)]

    def passes(self, seat: Seat) -> list[Event]:
        """This seat does nothing this turn."""
        event = Passed(seat=seat, seq=self._tick())
        self._emit(event)
        return [event]

    # ----------------------------------------------------------- internals

    def _validate(
        self,
        seat: Seat,
        suit: Suit,
        side: Side,
        price: Price,
        quantity: Quantity,
    ) -> RejectReason | None:
        """Pre-trade risk checks. Reject before matching, never after."""
        if not 0 <= seat < self.n_seats:
            raise ValueError(f"no such seat: {seat}")
        if price <= 0:
            return RejectReason.INVALID_PRICE
        if quantity <= 0:
            return RejectReason.INVALID_QUANTITY
        if side is Side.SELL and self.available_cards(seat, suit) < quantity:
            return RejectReason.NO_INVENTORY
        if side is Side.BUY and self.available_cash(seat) < price * quantity:
            return RejectReason.INSUFFICIENT_CASH
        return None

    def _transfer(
        self, buyer: Seat, seller: Seat, suit: Suit, quantity: int, price: int
    ) -> None:
        """Move cards one way and cash the other. The only mutation of accounts."""
        self._hands[seller][suit] -= quantity
        self._hands[buyer][suit] += quantity
        self._cash[buyer] -= price * quantity
        self._cash[seller] += price * quantity

        if self._hands[seller][suit] < 0:
            raise AssertionError(f"seat {seller} went short {suit.name}")
        if self._cash[buyer] < 0:
            raise AssertionError(f"seat {buyer} went cash-negative")

    def _sweep(self) -> list[Event]:
        """Figgie's post-trade rule: every quote in every suit is cancelled.

        One event per order rather than a single opaque "book wiped", so replay
        can reconstruct the book and the opponent model can tell an involuntary
        sweep from a voluntary cancel.
        """
        events: list[Event] = []
        for suit, book in self._books.items():
            for order_id in book.clear():
                event = OrderCancelled(
                    order_id=order_id,
                    seat=book._orders[order_id].seat,
                    suit=suit,
                    reason=CancelReason.TRADE_SWEEP,
                    seq=self._tick(),
                )
                self._emit(event)
                events.append(event)
        return events

    def _reject_cancel(self, seat: Seat, reason: RejectReason) -> OrderRejected:
        event = OrderRejected(
            seat=seat,
            suit=Suit.SPADES,
            side=Side.BUY,
            price=Price(1),
            quantity=Quantity(1),
            reason=reason,
            seq=self._tick(),
        )
        self._emit(event)
        return event

    # ------------------------------------------------------------- checks

    def assert_invariants(self) -> None:
        """Cheap runtime conservation check. Call it from tests and the loop."""
        for suit in Suit:
            held = sum(hand[suit] for hand in self._hands)
            if held != self.deal.composition[suit]:
                raise AssertionError(
                    f"card conservation broken for {suit.name}: "
                    f"{held} vs {self.deal.composition[suit]}"
                )
        total = sum(self._cash)
        if total != self.starting_cash * self.n_seats:
            raise AssertionError(
                f"cash conservation broken: {total} vs "
                f"{self.starting_cash * self.n_seats}"
            )
        if any(c < 0 for hand in self._hands for c in hand):
            raise AssertionError("negative holdings")

    def __str__(self) -> str:
        quotes = "  ".join(str(self._books[s]) for s in Suit)
        return f"[{quotes}]  cash={self.cash()}"