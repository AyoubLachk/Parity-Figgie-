"""Exchange tests.

The book's invariants were about order priority. These are about *money and
cards*, which is where a bug costs you the whole tournament: a slightly wrong
PnL looks exactly like a slightly better agent.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from parity.dealer import Deal, deal
from parity.events import (
    DeckRevealed,
    HandDealt,
    OrderCancelled,
    OrderPosted,
    OrderRejected,
    Traded,
    redact,
)
from parity.exchange import Exchange
from parity.orders import (
    CancelReason,
    OrderId,
    Price,
    RejectReason,
    Seat,
    Side,
)
from parity.rng import game_streams
from parity.settlement import settle
from parity.types import POT, Suit


@pytest.fixture
def a_deal() -> Deal:
    return deal(game_streams(2026, 0, n_agents=4).deal)


@pytest.fixture
def ex(a_deal: Deal) -> Exchange:
    return Exchange(deal=a_deal)


def _seat_holding(ex: Exchange, suit: Suit, at_least: int = 1) -> Seat:
    """A seat that actually holds `at_least` cards of `suit`."""
    for i, hand in enumerate(ex.hands()):
        if hand[suit] >= at_least:
            return Seat(i)
    pytest.skip(f"no seat holds {at_least} of {suit.name}")


# ------------------------------------------------------------------- setup


def test_opens_with_the_dealt_hands(ex: Exchange, a_deal: Deal) -> None:
    assert ex.hands() == a_deal.hands
    assert ex.cash() == (350,) * 4
    ex.assert_invariants()


def test_books_start_empty(ex: Exchange) -> None:
    for suit in Suit:
        assert ex.book(suit).best_bid() is None
        assert ex.book(suit).best_ask() is None


def test_the_log_opens_with_ground_truth(ex: Exchange) -> None:
    assert any(isinstance(e, DeckRevealed) for e in ex.log)
    assert sum(isinstance(e, HandDealt) for e in ex.log) == 4


# -------------------------------------------------------------- validation


def test_cannot_sell_a_card_you_do_not_hold(ex: Exchange) -> None:
    seat = next(
        Seat(i) for i, h in enumerate(ex.hands()) if h[Suit.SPADES] == 0
    ) if any(h[Suit.SPADES] == 0 for h in ex.hands()) else None
    if seat is None:
        pytest.skip("every seat holds spades in this deal")

    events = ex.submit(seat, Suit.SPADES, Side.SELL, Price(5))

    assert isinstance(events[0], OrderRejected)
    assert events[0].reason is RejectReason.NO_INVENTORY


def test_cannot_buy_above_your_cash(ex: Exchange) -> None:
    events = ex.submit(Seat(0), Suit.CLUBS, Side.BUY, Price(9_999))

    assert isinstance(events[0], OrderRejected)
    assert events[0].reason is RejectReason.INSUFFICIENT_CASH


def test_resting_sells_reserve_the_cards(ex: Exchange) -> None:
    """One card cannot back two sell orders."""
    seat = _seat_holding(ex, Suit.HEARTS, at_least=1)
    held = ex.hands()[seat][Suit.HEARTS]

    for _ in range(held):
        ex.submit(seat, Suit.HEARTS, Side.SELL, Price(5))

    assert ex.available_cards(seat, Suit.HEARTS) == 0
    events = ex.submit(seat, Suit.HEARTS, Side.SELL, Price(5))
    assert isinstance(events[0], OrderRejected)
    assert events[0].reason is RejectReason.NO_INVENTORY


def test_resting_buys_reserve_the_cash(ex: Exchange) -> None:
    ex.submit(Seat(0), Suit.CLUBS, Side.BUY, Price(300))
    assert ex.available_cash(Seat(0)) == 50

    events = ex.submit(Seat(0), Suit.HEARTS, Side.BUY, Price(100))
    assert isinstance(events[0], OrderRejected)
    assert events[0].reason is RejectReason.INSUFFICIENT_CASH


@pytest.mark.parametrize("price", [0, -5])
def test_nonsense_prices_are_rejected(ex: Exchange, price: int) -> None:
    events = ex.submit(Seat(0), Suit.CLUBS, Side.BUY, Price(price))
    assert isinstance(events[0], OrderRejected)
    assert events[0].reason is RejectReason.INVALID_PRICE


def test_rejections_are_events_not_exceptions(ex: Exchange) -> None:
    """Agents will send garbage. That must be countable, not fatal."""
    before = len(ex.log)
    ex.submit(Seat(0), Suit.CLUBS, Side.BUY, Price(9_999))
    assert len(ex.log) == before + 1
    ex.assert_invariants()


# ---------------------------------------------------------------- trading


def test_a_trade_moves_a_card_and_the_cash(ex: Exchange) -> None:
    seller = _seat_holding(ex, Suit.DIAMONDS)
    buyer = Seat((seller + 1) % 4)

    cards_before = ex.hands()[buyer][Suit.DIAMONDS]
    cash_before = ex.cash()

    ex.submit(seller, Suit.DIAMONDS, Side.SELL, Price(7))
    events = ex.submit(buyer, Suit.DIAMONDS, Side.BUY, Price(7))

    trades = [e for e in events if isinstance(e, Traded)]
    assert len(trades) == 1

    assert ex.hands()[buyer][Suit.DIAMONDS] == cards_before + 1
    assert ex.cash()[buyer] == cash_before[buyer] - 7
    assert ex.cash()[seller] == cash_before[seller] + 7
    ex.assert_invariants()


def test_trade_price_is_the_resting_price(ex: Exchange) -> None:
    seller = _seat_holding(ex, Suit.CLUBS)
    buyer = Seat((seller + 1) % 4)

    ex.submit(seller, Suit.CLUBS, Side.SELL, Price(4))
    events = ex.submit(buyer, Suit.CLUBS, Side.BUY, Price(9))

    trade = next(e for e in events if isinstance(e, Traded))
    assert trade.trade.price == 4


def test_a_trade_sweeps_every_book(ex: Exchange) -> None:
    """Figgie's defining microstructure rule."""
    seller = _seat_holding(ex, Suit.CLUBS)
    other = Seat((seller + 1) % 4)

    # Quotes resting in unrelated suits.
    ex.submit(other, Suit.SPADES, Side.BUY, Price(3))
    ex.submit(other, Suit.HEARTS, Side.BUY, Price(3))
    assert ex.book(Suit.SPADES).best_bid() is not None

    ex.submit(seller, Suit.CLUBS, Side.SELL, Price(5))
    ex.submit(other, Suit.CLUBS, Side.BUY, Price(5))

    for suit in Suit:
        assert ex.book(suit).best_bid() is None, f"{suit.name} not swept"
        assert ex.book(suit).best_ask() is None


def test_the_sweep_is_logged_per_order(ex: Exchange) -> None:
    seller = _seat_holding(ex, Suit.CLUBS)
    other = Seat((seller + 1) % 4)

    ex.submit(other, Suit.SPADES, Side.BUY, Price(3))
    ex.submit(seller, Suit.CLUBS, Side.SELL, Price(5))
    ex.submit(other, Suit.CLUBS, Side.BUY, Price(5))

    swept = [
        e
        for e in ex.log
        if isinstance(e, OrderCancelled) and e.reason is CancelReason.TRADE_SWEEP
    ]
    assert len(swept) == 1
    assert swept[0].suit is Suit.SPADES


def test_general_mode_leaves_the_book_alone(a_deal: Deal) -> None:
    """The same engine, run as a conventional CDA."""
    ex = Exchange(deal=a_deal, cancel_all_on_trade=False)
    seller = _seat_holding(ex, Suit.CLUBS)
    other = Seat((seller + 1) % 4)

    ex.submit(other, Suit.SPADES, Side.BUY, Price(3))
    ex.submit(seller, Suit.CLUBS, Side.SELL, Price(5))
    ex.submit(other, Suit.CLUBS, Side.BUY, Price(5))

    assert ex.book(Suit.SPADES).best_bid() is not None


# -------------------------------------------------------------- cancelling


def test_cancel_removes_your_order(ex: Exchange) -> None:
    events = ex.submit(Seat(0), Suit.CLUBS, Side.BUY, Price(4))
    posted = next(e for e in events if isinstance(e, OrderPosted))

    out = ex.cancel(Seat(0), posted.order.order_id)

    assert isinstance(out[0], OrderCancelled)
    assert out[0].reason is CancelReason.EXPLICIT
    assert ex.book(Suit.CLUBS).best_bid() is None


def test_cannot_cancel_someone_elses_order(ex: Exchange) -> None:
    events = ex.submit(Seat(0), Suit.CLUBS, Side.BUY, Price(4))
    posted = next(e for e in events if isinstance(e, OrderPosted))

    out = ex.cancel(Seat(1), posted.order.order_id)

    assert isinstance(out[0], OrderRejected)
    assert out[0].reason is RejectReason.NOT_YOUR_ORDER
    assert ex.book(Suit.CLUBS).best_bid() is not None


def test_cancelling_an_unknown_order_is_rejected(ex: Exchange) -> None:
    out = ex.cancel(Seat(0), OrderId(999))
    assert isinstance(out[0], OrderRejected)
    assert out[0].reason is RejectReason.UNKNOWN_ORDER


def test_cancelling_frees_the_reservation(ex: Exchange) -> None:
    events = ex.submit(Seat(0), Suit.CLUBS, Side.BUY, Price(300))
    posted = next(e for e in events if isinstance(e, OrderPosted))
    assert ex.available_cash(Seat(0)) == 50

    ex.cancel(Seat(0), posted.order.order_id)
    assert ex.available_cash(Seat(0)) == 350


# ------------------------------------------------------------- redaction


def test_agents_cannot_see_the_deck(ex: Exchange) -> None:
    """The single function standing between an agent and cheating."""
    view = redact(ex.log, Seat(0))
    assert not any(isinstance(e, DeckRevealed) for e in view)


def test_agents_see_only_their_own_hand(ex: Exchange) -> None:
    view = redact(ex.log, Seat(2))
    dealt = [e for e in view if isinstance(e, HandDealt)]
    assert len(dealt) == 1
    assert dealt[0].seat == 2
    assert dealt[0].hand == ex.deal.hands[2]


def test_trades_stay_public(ex: Exchange) -> None:
    """Who traded with whom is public in Figgie -- the opponent model needs it."""
    seller = _seat_holding(ex, Suit.CLUBS)
    buyer = Seat((seller + 1) % 4)
    ex.submit(seller, Suit.CLUBS, Side.SELL, Price(5))
    ex.submit(buyer, Suit.CLUBS, Side.BUY, Price(5))

    for seat in range(4):
        view = redact(ex.log, Seat(seat))
        trades = [e for e in view if isinstance(e, Traded)]
        assert len(trades) == 1
        assert trades[0].trade.buyer == buyer
        assert trades[0].trade.seller == seller


# ----------------------------------------------------- settles at the end


def test_hands_after_trading_still_settle_to_the_pot(ex: Exchange) -> None:
    seller = _seat_holding(ex, Suit.CLUBS)
    buyer = Seat((seller + 1) % 4)
    ex.submit(seller, Suit.CLUBS, Side.SELL, Price(5))
    ex.submit(buyer, Suit.CLUBS, Side.BUY, Price(5))

    payouts = settle(ex.deal.composition, ex.hands())
    assert sum(payouts) == POT


# ------------------------------------------------- the state machine


class ExchangeMachine(RuleBasedStateMachine):
    """Random order flow. Cards and cash must survive all of it."""

    def __init__(self) -> None:
        super().__init__()
        self.deal = deal(np.random.default_rng(7))
        self.ex = Exchange(deal=self.deal)
        self.posted: list[OrderId] = []

    @rule(
        seat=st.integers(0, 3),
        suit=st.sampled_from(Suit),
        side=st.sampled_from(Side),
        price=st.integers(1, 15),
    )
    def submit(self, seat: int, suit: Suit, side: Side, price: int) -> None:
        for event in self.ex.submit(Seat(seat), suit, side, Price(price)):
            if isinstance(event, OrderPosted):
                self.posted.append(event.order.order_id)

    @precondition(lambda self: bool(self.posted))
    @rule(seat=st.integers(0, 3), idx=st.integers(0, 60))
    def cancel(self, seat: int, idx: int) -> None:
        self.ex.cancel(Seat(seat), self.posted[idx % len(self.posted)])

    # --------------------------------------------------------- invariants

    @invariant()
    def conservation_holds(self) -> None:
        self.ex.assert_invariants()

    @invariant()
    def nobody_is_short(self) -> None:
        for hand in self.ex.hands():
            assert all(c >= 0 for c in hand)

    @invariant()
    def nobody_is_overdrawn(self) -> None:
        assert all(c >= 0 for c in self.ex.cash())

    @invariant()
    def reservations_never_exceed_holdings(self) -> None:
        for seat in range(4):
            for suit in Suit:
                assert self.ex.available_cards(Seat(seat), suit) >= 0
            assert self.ex.available_cash(Seat(seat)) >= 0

    @invariant()
    def no_book_is_crossed(self) -> None:
        for suit in Suit:
            assert not self.ex.book(suit).is_crossed()

    @invariant()
    def settlement_always_balances(self) -> None:
        assert sum(settle(self.deal.composition, self.ex.hands())) == POT

    @invariant()
    def no_seat_holds_more_than_exists(self) -> None:
        for hand in self.ex.hands():
            for suit in Suit:
                assert hand[suit] <= self.deal.composition[suit]


TestExchangeInvariants = ExchangeMachine.TestCase
TestExchangeInvariants.settings = settings(
    max_examples=150,
    stateful_step_count=30,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)