"""A price-time priority order book for one instrument.

One book per suit; the exchange owns four of them.

DATA STRUCTURE
--------------
Two binary heaps (`heapq`), one per side, holding (price_key, seq, order_id):

  * asks use price_key = +price, so the min-heap top is the cheapest offer;
  * bids use price_key = -price, so the min-heap top is the highest bid.

`seq` is a monotonic counter, so ties at the same price break by arrival order.
That single tuple encodes price-time priority exactly.

Heaps give O(1) "best price" and O(log n) insert, but they cannot remove an
arbitrary element. Hence LAZY DELETION: cancelling sets the order's remaining
quantity to 0 in a dict and leaves the heap entry in place. Every read pops dead
entries off the top first. Amortised cost is fine and the code stays small.

FIGGIE VS A GENERAL EXCHANGE
---------------------------
Figgie wipes every quote in every suit after each trade. That is the exchange's
job (it must sweep all four books), so this class exposes `clear()` and stays
agnostic. Likewise Figgie orders are always size 1, but partial fills are
supported so the same engine works as a conventional CDA -- and so this remains
an honest rehearsal for the C++ version.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from parity.orders import Order, OrderId, Quantity, Seat, Side, Trade
from parity.types import Suit

# (price_key, seq, order_id) -- the tuple that encodes price-time priority.
_Entry = tuple[int, int, OrderId]


@dataclass(slots=True)
class OrderBook:
    """The resting orders for one suit."""

    suit: Suit

    _bids: list[_Entry] = field(default_factory=list)
    _asks: list[_Entry] = field(default_factory=list)
    _orders: dict[OrderId, Order] = field(default_factory=dict)
    _remaining: dict[OrderId, int] = field(default_factory=dict)

    #: Quantity removed by self-trade prevention. Diagnostics, and a term
    #: the conservation invariant needs to balance.
    stp_cancelled: int = 0

    # ------------------------------------------------------------ internals

    def _heap(self, side: Side) -> list[_Entry]:
        return self._bids if side is Side.BUY else self._asks

    def _is_live(self, order_id: OrderId) -> bool:
        return self._remaining.get(order_id, 0) > 0

    def _prune(self, heap: list[_Entry]) -> None:
        """Discard dead entries from the top. The lazy half of lazy deletion."""
        while heap and not self._is_live(heap[0][2]):
            heapq.heappop(heap)

    def _peek(self, side: Side) -> Order | None:
        heap = self._heap(side)
        self._prune(heap)
        return self._orders[heap[0][2]] if heap else None

    # --------------------------------------------------------------- reads

    def best_bid(self) -> Order | None:
        """Highest-priced live buy order, or None."""
        return self._peek(Side.BUY)

    def best_ask(self) -> Order | None:
        """Lowest-priced live sell order, or None."""
        return self._peek(Side.SELL)

    def spread(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else ask.price - bid.price

    def remaining(self, order_id: OrderId) -> int:
        """Unfilled quantity. 0 for filled, cancelled, or unknown orders."""
        return self._remaining.get(order_id, 0)

    def live_orders(self) -> list[Order]:
        """Every resting order. For tests and for building agent views."""
        return [self._orders[oid] for oid in self._remaining if self._is_live(oid)]

    def total_resting(self) -> int:
        """Sum of unfilled quantity across the book. A conservation quantity."""
        return sum(q for q in self._remaining.values() if q > 0)

    def is_crossed(self) -> bool:
        """True if the best bid meets or exceeds the best ask.

        Must NEVER be true after add() returns -- add() matches until it cannot.
        """
        bid, ask = self.best_bid(), self.best_ask()
        return bid is not None and ask is not None and bid.price >= ask.price

    # -------------------------------------------------------------- writes

    def add(self, order: Order) -> list[Trade]:
        """Match `order` against the book, then rest whatever is left.

        Returns the trades produced, in execution order. The book is guaranteed
        uncrossed when this returns.

        Self-trade prevention: a resting order from the same seat is cancelled
        rather than traded against, and matching continues. The obvious
        alternative -- stop matching and rest the remainder -- leaves the book
        crossed against itself, which a stateful property test found in about
        three steps.
        """
        if order.suit is not self.suit:
            raise ValueError(f"{order.suit.name} order sent to {self.suit.name} book")
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order id {order.order_id}")

        self._orders[order.order_id] = order
        left = int(order.quantity)
        trades: list[Trade] = []
        opposite = self._heap(order.side.opposite)

        while left > 0:
            self._prune(opposite)
            if not opposite:
                break

            resting = self._orders[opposite[0][2]]

            if not self._crosses(order, resting):
                break
            if resting.seat == order.seat:
                # Self-trade prevention, 'cancel resting' policy. Stopping
                # here instead would leave the book CROSSED -- seat 0's own
                # bid at 9 resting above its own ask at 4. Cancelling the
                # resting side keeps the invariant and matches what CME and
                # other venues actually do.
                self.stp_cancelled += self._remaining[resting.order_id]
                self._remaining[resting.order_id] = 0
                continue

            traded = min(left, self._remaining[resting.order_id])
            self._remaining[resting.order_id] -= traded
            left -= traded

            buyer, seller = (
                (order.seat, resting.seat)
                if order.side is Side.BUY
                else (resting.seat, order.seat)
            )
            trades.append(
                Trade(
                    suit=self.suit,
                    price=resting.price,  # the passive side set the terms
                    quantity=Quantity(traded),
                    buyer=Seat(buyer),
                    seller=Seat(seller),
                    aggressor=order.side,
                    resting_order_id=resting.order_id,
                    incoming_order_id=order.order_id,
                    seq=order.seq,
                )
            )

        self._remaining[order.order_id] = left
        if left > 0:
            key = -order.price if order.side is Side.BUY else order.price
            heapq.heappush(
                self._heap(order.side), (int(key), order.seq, order.order_id)
            )

        return trades

    @staticmethod
    def _crosses(incoming: Order, resting: Order) -> bool:
        """Would these two trade? A buy crosses an ask at or below its limit."""
        if incoming.side is Side.BUY:
            return incoming.price >= resting.price
        return incoming.price <= resting.price

    def cancel(self, order_id: OrderId, seat: Seat | None = None) -> bool:
        """Cancel by tombstone. Returns False if it was not live.

        Pass `seat` to enforce ownership -- an agent cancelling someone else's
        order should fail rather than succeed quietly.
        """
        if not self._is_live(order_id):
            return False
        if seat is not None and self._orders[order_id].seat != seat:
            return False
        self._remaining[order_id] = 0
        return True

    def clear(self) -> list[OrderId]:
        """Cancel everything. Figgie's post-trade sweep.

        Returns the ids that were live, so the caller can log one cancel event
        per order rather than a single opaque "book wiped".
        """
        killed = [oid for oid in self._remaining if self._is_live(oid)]
        for oid in killed:
            self._remaining[oid] = 0
        self._bids.clear()
        self._asks.clear()
        return killed

    def __str__(self) -> str:
        bid, ask = self.best_bid(), self.best_ask()
        b = f"{bid.price}" if bid else "--"
        a = f"{ask.price}" if ask else "--"
        return f"{self.suit.name[:1]} {b} / {a}"