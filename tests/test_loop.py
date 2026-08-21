"""Game loop tests.

One test here matters more than the rest: PnL sums to zero. Everything the
project claims later -- "this agent beats that one by N chips" -- is only
meaningful if the game is genuinely zero-sum. An agent that appears to profit
from nothing looks exactly like an agent that is subtly better.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from parity.actions import Observation, Pass, PostOrder, Quote
from parity.agents import REGISTRY, Agent, HandAgent, RandomAgent, build
from parity.dealer import deal
from parity.events import DeckRevealed, HandDealt, Settled, Traded
from parity.loop import play
from parity.orders import Seat, Side
from parity.rng import game_streams
from parity.types import POT, Suit

seeds = st.integers(min_value=0, max_value=2**16)


def _game(names: list[str], seed: int = 2026, game: int = 0, turns: int = 120):
    streams = game_streams(seed, game, n_agents=len(names))
    d = deal(streams.deal, n_players=len(names))
    return d, play(d, [build(n) for n in names], event_budget=turns, streams=streams)


FOUR_RANDOM = ["random"] * 4
MIXED = ["partner", "hand", "random", "random"]


# ------------------------------------------------------- the control game


def test_four_passers_change_nothing() -> None:
    """No trades, so the final hands must equal the dealt hands exactly."""
    d, result = _game(["pass"] * 4)

    assert result.final_hands == d.hands
    assert not [e for e in result.log if isinstance(e, Traded)]
    assert sum(result.payouts) == POT
    assert sum(result.pnl) == 0


def test_a_game_with_trading_completes() -> None:
    _, result = _game(FOUR_RANDOM)
    assert [e for e in result.log if isinstance(e, Traded)], "nothing ever traded"
    assert isinstance(result.log[-1], Settled)


# ----------------------------------------------------- the headline property


@given(seed=seeds)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_pnl_is_zero_sum(seed: int) -> None:
    """The identity the whole project rests on.

        net_i = (final_cash_i - starting_cash) + payout_i - ante

    Trading only moves cash sideways, settlement distributes exactly POT, and
    the antes funded exactly POT. So the three terms cancel and the total is 0.
    """
    _, result = _game(MIXED, seed=seed, turns=60)
    assert sum(result.pnl) == 0
    assert sum(result.payouts) == POT


@given(seed=seeds)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_cards_survive_a_whole_game(seed: int) -> None:
    d, result = _game(FOUR_RANDOM, seed=seed, turns=60)
    for suit in Suit:
        held = sum(hand[suit] for hand in result.final_hands)
        assert held == d.composition[suit]
    assert all(c >= 0 for hand in result.final_hands for c in hand)


# ------------------------------------------------------------ determinism


def test_the_same_seed_replays_identically() -> None:
    _, a = _game(MIXED, seed=99)
    _, b = _game(MIXED, seed=99)

    assert a.pnl == b.pnl
    assert a.final_hands == b.final_hands
    assert len(a.log) == len(b.log)
    assert [str(e) for e in a.log] == [str(e) for e in b.log]


def test_different_seeds_give_different_games() -> None:
    results = [_game(MIXED, seed=2026, game=g)[1] for g in range(12)]
    assert len({r.pnl for r in results}) > 6


# ------------------------------------------------------------- redaction


class Spy(Agent):
    """Records everything it is told, so we can check it was told the right things."""

    name = "spy"

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[object] = []
        self.views: list[Observation] = []

    def on_event(self, event: object) -> None:  # type: ignore[override]
        self.seen.append(event)

    def act(self, view: Observation) -> Pass:
        self.views.append(view)
        return Pass()


def test_agents_are_never_told_the_deck() -> None:
    """If this ever fails, every result in the project is worthless."""
    streams = game_streams(2026, 0, n_agents=4)
    d = deal(streams.deal)
    spies = [Spy() for _ in range(4)]
    play(d, list(spies), event_budget=40, streams=streams)

    for spy in spies:
        assert not [e for e in spy.seen if isinstance(e, DeckRevealed)]


def test_each_agent_sees_only_its_own_hand() -> None:
    streams = game_streams(2026, 0, n_agents=4)
    d = deal(streams.deal)
    spies = [Spy() for _ in range(4)]
    play(d, list(spies), event_budget=40, streams=streams)

    for i, spy in enumerate(spies):
        dealt = [e for e in spy.seen if isinstance(e, HandDealt)]
        assert len(dealt) == 1
        assert dealt[0].seat == i
        assert dealt[0].hand == d.hands[i]


def test_the_observation_carries_no_hidden_state() -> None:
    """An Observation must not expose a route to the deck or other hands."""
    streams = game_streams(2026, 0, n_agents=4)
    d = deal(streams.deal)
    spies = [Spy() for _ in range(4)]
    play(d, list(spies), event_budget=20, streams=streams)

    view = spies[0].views[0]
    assert view.hand == d.hands[0]
    assert set(vars(Observation).get("__slots__", ())) == {
        "seat",
        "hand",
        "cash",
        "quotes",
        "my_orders",
        "events_remaining",
    }


# ---------------------------------------------------------------- agents


def test_every_registered_agent_can_play() -> None:
    for name in REGISTRY:
        if name == "spy":
            continue
        _, result = _game([name] * 4, turns=60)
        assert sum(result.pnl) == 0


def test_agents_use_only_their_own_rng() -> None:
    """Two runs with the same seed must produce byte-identical action streams."""
    a = _game(["random"] * 4, seed=5)[1]
    b = _game(["random"] * 4, seed=5)[1]
    assert a.pnl == b.pnl


def test_random_agent_only_offers_what_it_holds() -> None:
    agent = RandomAgent()
    agent.reset(Seat(0), (10, 0, 0, 0), 4, np.random.default_rng(3))
    view = Observation(
        seat=Seat(0),
        hand=(10, 0, 0, 0),
        cash=350,
        quotes={s: Quote(None, None)
                for s in Suit},
        my_orders=(),
        events_remaining=50,
    )
    for _ in range(200):
        action = agent.act(view)
        if isinstance(action, PostOrder) and action.side is Side.SELL:
            assert action.suit is Suit.SPADES


def test_hand_agent_targets_its_longest_suit() -> None:
    agent = HandAgent()
    agent.reset(Seat(0), (1, 6, 2, 1), 4, np.random.default_rng(0))
    view = Observation(
        seat=Seat(0),
        hand=(1, 6, 2, 1),
        cash=350,
        quotes={s: Quote(None, None)
                for s in Suit},
        my_orders=(),
        events_remaining=50,
    )
    buys = [
        a.suit
        for _ in range(300)
        if isinstance(a := agent.act(view), PostOrder) and a.side is Side.BUY
    ]
    assert buys and set(buys) == {Suit.CLUBS}


def test_partner_agent_targets_the_partner() -> None:
    """The one deduction: long clubs implies spades is the goal."""
    agent = build("partner")
    agent.reset(Seat(0), (1, 6, 2, 1), 4, np.random.default_rng(0))
    view = Observation(
        seat=Seat(0),
        hand=(1, 6, 2, 1),
        cash=350,
        quotes={s: Quote(None, None)
                for s in Suit},
        my_orders=(),
        events_remaining=50,
    )
    buys = [
        a.suit
        for _ in range(300)
        if isinstance(a := agent.act(view), PostOrder) and a.side is Side.BUY
    ]
    assert buys and set(buys) == {Suit.SPADES}


# ------------------------------------------------------------- guards


def test_wrong_number_of_agents_is_rejected() -> None:
    d = deal(np.random.default_rng(0))
    with pytest.raises(ValueError, match="agents"):
        play(d, [build("pass")] * 3)


def test_price_type_is_int_everywhere() -> None:
    """Floats in a book break price-time priority silently. Guard it."""
    _, result = _game(FOUR_RANDOM, turns=60)
    for event in result.log:
        if isinstance(event, Traded):
            assert isinstance(event.trade.price, int)
            assert not isinstance(event.trade.price, bool)