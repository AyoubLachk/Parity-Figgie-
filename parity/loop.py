"""The game loop.

Everything else in the project is a part. This is the thing that connects them,
and it is deliberately the smallest interesting file in the repo:

    deal -> loop { whose turn, what do they see, what do they do, apply it }
         -> settle -> PnL

THE PNL IDENTITY
----------------
    net_i = (final_cash_i - starting_cash) + payout_i - ante

Three components: what you made trading, what the pot paid you, what you put in.
And it sums to zero across seats, necessarily:

    sum(final_cash) == n * starting_cash   (trading only moves cash sideways)
    sum(payout)     == POT                 (settlement distributes the pot)
    sum(ante)       == POT                 (the antes funded the pot)

    => sum(net) == 0

That identity is `test_pnl_is_zero_sum`, and it is the single test that makes
every later result trustworthy. If PnL does not sum to zero, some agent appears
to profit from nothing -- and "appears to profit from nothing" is exactly what a
subtly better agent also looks like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from parity.actions import CancelOrder, Observation, Pass, PostOrder, Quote
from parity.agents import Agent
from parity.dealer import Deal
from parity.events import Event, GameEnded, Settled, redact
from parity.exchange import DEFAULT_STARTING_CASH, Exchange
from parity.orders import Seat
from parity.rng import GameStreams, game_streams
from parity.settlement import settle
from parity.types import POT, Suit, SuitCounts

DEFAULT_EVENT_BUDGET = 200


@dataclass(frozen=True, slots=True)
class GameResult:
    """One completed round."""

    log: tuple[Event, ...]
    final_hands: tuple[SuitCounts, ...]
    payouts: tuple[int, ...]
    pnl: tuple[int, ...]
    n_turns: int

    def winner(self) -> Seat:
        return Seat(max(range(len(self.pnl)), key=lambda i: self.pnl[i]))


def observe(exchange: Exchange, seat: Seat, turns_left: int) -> Observation:
    """Build the read-only view a seat is allowed to act on.

    Note what is absent: the deck composition, and every other seat's hand.
    Neither is reachable from here, which is the point -- an agent cannot cheat
    by accident because the information simply is not in the object.
    """
    quotes = {
        suit: Quote(
            bid=exchange.book(suit).best_bid(),
            ask=exchange.book(suit).best_ask(),
        )
        for suit in Suit
    }
    mine = tuple(
        order
        for suit in Suit
        for order in exchange.book(suit).live_orders()
        if order.seat == seat
    )
    return Observation(
        seat=seat,
        hand=exchange.hands()[seat],
        cash=exchange.cash()[seat],
        quotes=quotes,
        my_orders=mine,
        events_remaining=turns_left,
    )


def play(
    deal: Deal,
    agents: list[Agent],
    event_budget: int = DEFAULT_EVENT_BUDGET,
    streams: GameStreams | None = None,
    starting_cash: int = DEFAULT_STARTING_CASH,
    check_every: int = 0,
) -> GameResult:
    """Play one complete round of Figgie.

    `agents` maps onto seats by position: agents[0] sits in seat 0. The
    tournament harness rotates that mapping to cancel out positional effects.

    `check_every > 0` runs conservation checks that often -- useful while
    developing, since it fails at the step that broke rather than at the end.
    """
    n_seats = len(deal.hands)
    if len(agents) != n_seats:
        raise ValueError(f"{len(agents)} agents for {n_seats} seats")

    if streams is None:
        streams = game_streams(master_seed=0, game_index=0, n_agents=n_seats)

    exchange = Exchange(deal=deal, starting_cash=starting_cash)

    for i, agent in enumerate(agents):
        agent.reset(
            seat=Seat(i),
            hand=deal.hands[i],
            n_seats=n_seats,
            rng=streams.agents[i],
        )

    _broadcast(agents, exchange.log)

    for turn in range(event_budget):
        seat = Seat(turn % n_seats)
        view = observe(exchange, seat, turns_left=event_budget - turn)
        action = agents[seat].act(view)

        match action:
            case PostOrder(suit=suit, side=side, price=price):
                produced = exchange.submit(seat, suit, side, price)
            case CancelOrder(order_id=order_id):
                produced = exchange.cancel(seat, order_id)
            case Pass():
                produced = exchange.passes(seat)
            case _:
                assert_never(action)

        _broadcast(agents, produced)

        if check_every and turn % check_every == 0:
            exchange.assert_invariants()

    exchange.assert_invariants()

    final_hands = exchange.hands()
    payouts = settle(deal.composition, final_hands)
    ante = POT // n_seats
    pnl = tuple(
        (exchange.cash()[i] - starting_cash) + payouts[i] - ante
        for i in range(n_seats)
    )

    end = GameEnded(reason="event budget exhausted", seq=exchange._tick())
    exchange.log.append(end)
    result = Settled(
        final_hands=final_hands, payouts=payouts, seq=exchange._tick()
    )
    exchange.log.append(result)
    _broadcast(agents, [end, result])

    return GameResult(
        log=tuple(exchange.log),
        final_hands=final_hands,
        payouts=payouts,
        pnl=pnl,
        n_turns=event_budget,
    )


def _broadcast(agents: list[Agent], events: list[Event]) -> None:
    """Deliver events to every agent, redacted per seat.

    Two things that are easy to get wrong and expensive later:

      * REDACT. Handing agents the raw log shows them the deck composition and
        everyone's cards. `redact` is the one function standing in the way.
      * Send to EVERY agent, including whoever just acted. Your own fill is
        information -- somebody chose to trade against your price, which is
        usually bad news. That is the adverse-selection signal a serious agent
        needs, and dropping it here would quietly cap how good any agent can be.
    """
    for i, agent in enumerate(agents):
        for event in redact(events, Seat(i)):
            agent.on_event(event)