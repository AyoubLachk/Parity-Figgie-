"""Order book tests.

The example tests below check that specific things happen. The state machine at
the bottom checks that certain things NEVER happen, across thousands of random
add/cancel sequences Hypothesis invents. The second kind is what catches the
bugs that would otherwise surface on tournament day.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from parity.book import OrderBook
from parity.orders import Order, OrderId, Price, Quantity, Seat, Side
from parity.types import Suit

SUIT = Suit.CLUBS


class Maker:
    """Hands out orders with unique ids and increasing sequence numbers."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(
        self, seat: int, side: Side, price: int, qty: int = 1, suit: Suit = SUIT
    ) -> Order:
        self.n += 1
        return Order(
            order_id=OrderId(self.n),
            seat=Seat(seat),
            suit=suit,
            side=side,
            price=Price(price),
            quantity=Quantity(qty),
            seq=self.n,
        )


@pytest.fixture
def make() -> Maker:
    return Maker()


@pytest.fixture
def book() -> OrderBook:
    return OrderBook(suit=SUIT)


# ------------------------------------------------------------------ resting


def test_empty_book_has_no_best_prices(book: OrderBook) -> None:
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.spread() is None
    assert book.total_resting() == 0


def test_non_crossing_order_rests(book: OrderBook, make: Maker) -> None:
    o = make(seat=0, side=Side.BUY, price=5)
    assert book.add(o) == []
    assert book.best_bid() == o
    assert book.remaining(o.order_id) == 1


def test_best_bid_is_the_highest(book: OrderBook, make: Maker) -> None:
    book.add(make(0, Side.BUY, 4))
    top = make(1, Side.BUY, 7)
    book.add(top)
    book.add(make(2, Side.BUY, 6))
    assert book.best_bid() == top


def test_best_ask_is_the_lowest(book: OrderBook, make: Maker) -> None:
    book.add(make(0, Side.SELL, 9))
    top = make(1, Side.SELL, 6)
    book.add(top)
    book.add(make(2, Side.SELL, 8))
    assert book.best_ask() == top


def test_time_priority_breaks_price_ties(book: OrderBook, make: Maker) -> None:
    """Same price: whoever arrived first trades first."""
    first = make(0, Side.BUY, 5)
    second = make(1, Side.BUY, 5)
    book.add(first)
    book.add(second)
    assert book.best_bid() == first

    trades = book.add(make(2, Side.SELL, 5))
    assert len(trades) == 1
    assert trades[0].resting_order_id == first.order_id


# ----------------------------------------------------------------- matching


def test_a_crossing_order_trades(book: OrderBook, make: Maker) -> None:
    resting = make(0, Side.SELL, 5)
    book.add(resting)

    trades = book.add(make(1, Side.BUY, 7))

    assert len(trades) == 1
    t = trades[0]
    assert t.price == 5, "trade happens at the RESTING order's price"
    assert t.buyer == 1
    assert t.seller == 0
    assert t.aggressor is Side.BUY
    assert book.total_resting() == 0


def test_the_aggressor_is_whoever_crossed(book: OrderBook, make: Maker) -> None:
    book.add(make(0, Side.BUY, 8))
    trades = book.add(make(1, Side.SELL, 3))
    assert trades[0].aggressor is Side.SELL
    assert trades[0].price == 8


def test_matching_walks_the_book(book: OrderBook, make: Maker) -> None:
    book.add(make(0, Side.SELL, 4))
    book.add(make(1, Side.SELL, 5))
    book.add(make(2, Side.SELL, 6))

    trades = book.add(make(3, Side.BUY, 5, qty=3))

    assert [t.price for t in trades] == [4, 5], "cheapest first, stops above limit"
    assert book.best_ask() is not None
    assert book.best_ask().price == 6  # type: ignore[union-attr]


def test_partial_fill_rests_the_remainder(book: OrderBook, make: Maker) -> None:
    book.add(make(0, Side.SELL, 5, qty=2))
    incoming = make(1, Side.BUY, 5, qty=5)

    trades = book.add(incoming)

    assert len(trades) == 1
    assert trades[0].quantity == 2
    assert book.remaining(incoming.order_id) == 3
    assert book.best_bid() == incoming


def test_book_is_never_crossed_after_add(book: OrderBook, make: Maker) -> None:
    book.add(make(0, Side.BUY, 6))
    book.add(make(1, Side.SELL, 4))
    assert not book.is_crossed()


def test_self_trade_prevention_cancels_the_resting_order(
    book: OrderBook, make: Maker
) -> None:
    """Seat 0 crossing its own quote kills the resting side, and never trades.

    The tempting alternative -- stop matching, rest the remainder -- leaves
    seat 0 bidding 9 while its own ask sits at 4. A crossed book. The stateful
    test below found that in three steps.
    """
    mine = make(0, Side.SELL, 4)
    book.add(mine)

    incoming = make(0, Side.BUY, 9)
    trades = book.add(incoming)

    assert trades == [], "no self-trade happened"
    assert book.remaining(mine.order_id) == 0, "the resting order was cancelled"
    assert book.best_bid() == incoming
    assert not book.is_crossed()


# --------------------------------------------------------------- cancelling


def test_cancel_removes_from_the_book(book: OrderBook, make: Maker) -> None:
    o = make(0, Side.BUY, 5)
    book.add(o)

    assert book.cancel(o.order_id) is True
    assert book.best_bid() is None
    assert book.remaining(o.order_id) == 0


def test_a_cancelled_order_never_trades(book: OrderBook, make: Maker) -> None:
    o = make(0, Side.SELL, 5)
    book.add(o)
    book.cancel(o.order_id)

    assert book.add(make(1, Side.BUY, 9)) == []


def test_cancelling_twice_returns_false(book: OrderBook, make: Maker) -> None:
    o = make(0, Side.BUY, 5)
    book.add(o)
    assert book.cancel(o.order_id) is True
    assert book.cancel(o.order_id) is False


def test_cannot_cancel_someone_elses_order(book: OrderBook, make: Maker) -> None:
    o = make(0, Side.BUY, 5)
    book.add(o)

    assert book.cancel(o.order_id, seat=Seat(1)) is False
    assert book.cancel(o.order_id, seat=Seat(0)) is True


def test_cancelling_an_unknown_order_returns_false(book: OrderBook) -> None:
    assert book.cancel(OrderId(999)) is False


def test_clear_wipes_both_sides(book: OrderBook, make: Maker) -> None:
    """Figgie's post-trade sweep."""
    a = make(0, Side.BUY, 4)
    b = make(1, Side.SELL, 8)
    book.add(a)
    book.add(b)

    killed = book.clear()

    assert set(killed) == {a.order_id, b.order_id}
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.total_resting() == 0


# ------------------------------------------------------------------ guards


def test_wrong_suit_is_rejected(book: OrderBook, make: Maker) -> None:
    with pytest.raises(ValueError, match="book"):
        book.add(make(0, Side.BUY, 5, suit=Suit.HEARTS))


def test_duplicate_order_id_is_rejected(book: OrderBook, make: Maker) -> None:
    o = make(0, Side.BUY, 5)
    book.add(o)
    with pytest.raises(ValueError, match="duplicate"):
        book.add(o)


@pytest.mark.parametrize(("price", "qty"), [(0, 1), (-3, 1), (5, 0), (5, -2)])
def test_nonsense_orders_are_rejected(price: int, qty: int) -> None:
    with pytest.raises(ValueError):
        Order(OrderId(1), Seat(0), SUIT, Side.BUY, Price(price), Quantity(qty), 1)


# ------------------------------------------------- the state machine
#
# Everything above says "this specific thing happens". The machine below says
# "these things never happen, no matter what sequence of operations you throw at
# it". Hypothesis generates the sequences, and shrinks any failure to the
# shortest one that still breaks. This is the highest-value test in the file.


class BookMachine(RuleBasedStateMachine):
    """Fire random adds and cancels; assert the invariants after every step."""

    def __init__(self) -> None:
        super().__init__()
        self.book = OrderBook(suit=SUIT)
        self.make = Maker()
        self.live: list[OrderId] = []
        self.added = 0
        self.cancelled = 0
        self.filled = 0

    @rule(
        seat=st.integers(0, 3),
        side=st.sampled_from(Side),
        price=st.integers(1, 12),
        qty=st.integers(1, 4),
    )
    def add_order(self, seat: int, side: Side, price: int, qty: int) -> None:
        order = self.make(seat, side, price, qty)
        stp_before = self.book.stp_cancelled
        trades = self.book.add(order)

        self.added += qty
        self.filled += sum(int(t.quantity) for t in trades) * 2  # both sides
        self.cancelled += self.book.stp_cancelled - stp_before
        self.live.append(order.order_id)

    @precondition(lambda self: bool(self.live))
    @rule(idx=st.integers(0, 50))
    def cancel_order(self, idx: int) -> None:
        oid = self.live[idx % len(self.live)]
        before = self.book.remaining(oid)
        if self.book.cancel(oid):
            self.cancelled += before

    @precondition(lambda self: bool(self.live))
    @rule()
    def sweep(self) -> None:
        # Read before clearing -- clear() zeroes the remainders on its way out.
        self.cancelled += self.book.total_resting()
        self.book.clear()

    # --------------------------------------------------------- invariants

    @invariant()
    def never_crossed(self) -> None:
        """The defining property of a matching engine."""
        assert not self.book.is_crossed()

    @invariant()
    def best_prices_are_actually_best(self) -> None:
        live = self.book.live_orders()
        bids = [o.price for o in live if o.side is Side.BUY]
        asks = [o.price for o in live if o.side is Side.SELL]

        bid, ask = self.book.best_bid(), self.book.best_ask()
        assert (bid.price if bid else None) == (max(bids) if bids else None)
        assert (ask.price if ask else None) == (min(asks) if asks else None)

    @invariant()
    def quantity_is_conserved(self) -> None:
        """added = still resting + cancelled + filled. Nothing appears or vanishes."""
        assert self.book.total_resting() == self.added - self.cancelled - self.filled

    @invariant()
    def no_negative_remainders(self) -> None:
        assert all(q >= 0 for q in self.book._remaining.values())

    @invariant()
    def every_live_order_is_findable(self) -> None:
        for order in self.book.live_orders():
            assert self.book.remaining(order.order_id) > 0
            assert order.suit is SUIT


TestBookInvariants = BookMachine.TestCase
TestBookInvariants.settings = settings(
    max_examples=200,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)