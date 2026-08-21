"""Agents.

An abstract base plus a registry, so adding an agent is exactly one new class
with a `@register` decorator and a `name`, and the command line can then refer
to it by string.

Four agents ship here, and the weak ones are not filler -- you cannot claim an
agent is good without something for it to be good *against*:

    PassAgent     does nothing.        The control.
    RandomAgent   trades at random.    The floor.
    HandAgent     buys what it holds most of.       Naive, and should LOSE.
    PartnerAgent  buys the partner of what it holds most of.  Should WIN.

The last two are the whole intellectual content of Figgie in miniature. Holding
a lot of clubs is evidence that clubs is the 12-card *common* suit -- which pays
nothing -- and therefore that spades, its same-colour partner, is the goal.
HandAgent misses that inversion. PartnerAgent makes exactly one deduction and
should beat it by a measurable margin. That gap is a result you can report.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from parity.actions import Action, Observation, Pass, PostOrder
from parity.events import Event
from parity.orders import Price, Seat, Side
from parity.rng import Generator
from parity.types import Suit, SuitCounts

MIN_PRICE = 1
MAX_PRICE = 15


class Agent(ABC):
    """One player.

    Lifecycle, per game:
        reset(...)          once, before anything happens
        on_event(event)     for every public event, in order
        act(view)           when it is this seat's turn

    An agent holds state between calls -- a posterior is state -- but `reset`
    must wipe it completely, so a game is fully determined by the deal plus the
    seeds and nothing leaks between games.
    """

    name: ClassVar[str] = "abstract"

    def __init__(self) -> None:
        self.seat: Seat = Seat(0)
        self.hand: SuitCounts = (0, 0, 0, 0)
        self.n_seats: int = 4
        self.rng: Generator | None = None

    def reset(
        self, seat: Seat, hand: SuitCounts, n_seats: int, rng: Generator
    ) -> None:
        """Start a fresh game. Subclasses that add state must extend this."""
        self.seat = seat
        self.hand = hand
        self.n_seats = n_seats
        self.rng = rng

    def on_event(self, event: Event) -> None:
        """A public event happened. Default: ignore it."""

    @abstractmethod
    def act(self, view: Observation) -> Action:
        """Your turn. Return exactly one action."""

    def _uniform_price(self) -> Price:
        """A price drawn from the agent's own stream.

        Never `random.randint`. Global RNG state is invisible, order-dependent,
        and does not survive multiprocessing -- and it would silently break the
        common random numbers the whole tournament design rests on.
        """
        assert self.rng is not None, "act() called before reset()"
        return Price(int(self.rng.integers(MIN_PRICE, MAX_PRICE + 1)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seat={self.seat})"


# ----------------------------------------------------------------- registry

REGISTRY: dict[str, type[Agent]] = {}


def register(cls: type[Agent]) -> type[Agent]:
    """Add an agent under its `name`. One decorator, and `--agents random` works."""
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate agent name: {cls.name}")
    REGISTRY[cls.name] = cls
    return cls


def build(name: str) -> Agent:
    if name not in REGISTRY:
        raise KeyError(f"unknown agent {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]()


# ------------------------------------------------------------------ agents


@register
class PassAgent(Agent):
    """Never trades.

    The control. A game of four of these must still deal, settle, and pay out
    exactly the pot -- if that fails, the bug is in the loop, not the agents.
    """

    name = "pass"

    def act(self, view: Observation) -> Action:
        return Pass()


@register
class RandomAgent(Agent):
    """Uniform noise, constrained to be mostly legal.

    Only offers suits it actually holds, and passes a third of the time --
    otherwise it floods the book with rejections and nothing interesting ever
    trades. Still has no idea what anything is worth.
    """

    name = "random"

    pass_probability: ClassVar[float] = 0.3

    def act(self, view: Observation) -> Action:
        assert self.rng is not None
        if self.rng.random() < self.pass_probability:
            return Pass()

        sellable = [s for s in Suit if view.hand[s] > 0]
        side = Side.BUY if not sellable or self.rng.random() < 0.5 else Side.SELL

        if side is Side.SELL:
            suit = sellable[int(self.rng.integers(len(sellable)))]
        else:
            suit = Suit(int(self.rng.integers(4)))

        return PostOrder(suit=suit, side=side, price=self._uniform_price())


@register
class HandAgent(Agent):
    """Buys the suit it holds most of. Deliberately wrong.

    The reasoning is the obvious one: "I have five clubs, clubs must be good."
    It is exactly backwards. Five clubs is roughly 14-to-1 evidence that clubs
    is the 12-card suit rather than the 8-card one, and the 12-card suit is the
    common suit, which pays nothing.

    Kept as a baseline precisely because it fails for a reason you can state.
    """

    name = "hand"

    def act(self, view: Observation) -> Action:
        assert self.rng is not None
        if self.rng.random() < 0.2:
            return Pass()

        target = view.most_held()
        quote = view.quotes[target]

        # Lift a cheap offer if there is one, else bid modestly.
        if quote.ask_price is not None and quote.ask_price <= 8:
            return PostOrder(target, Side.BUY, Price(quote.ask_price))

        dump = view.least_held()
        if view.hand[dump] > 0 and self.rng.random() < 0.5:
            return PostOrder(dump, Side.SELL, Price(int(self.rng.integers(6, 12))))

        return PostOrder(target, Side.BUY, Price(int(self.rng.integers(3, 8))))


@register
class PartnerAgent(Agent):
    """Buys the same-colour partner of the suit it holds most of.

    One inference step, and the one that matters. A long holding in clubs says
    clubs is probably the common suit, so spades -- its partner -- is probably
    the goal suit. So sell clubs and buy spades.

    Structurally identical to HandAgent, targeting `.partner` instead. If the
    inversion is real, this beats it. If it does not, either the inversion is
    wrong or the harness is.
    """

    name = "partner"

    def act(self, view: Observation) -> Action:
        assert self.rng is not None
        if self.rng.random() < 0.2:
            return Pass()

        common_guess = view.most_held()
        target = common_guess.partner  # the goal-suit guess
        quote = view.quotes[target]

        if quote.ask_price is not None and quote.ask_price <= 10:
            return PostOrder(target, Side.BUY, Price(quote.ask_price))

        # The suit we hold most of is probably worthless. Offer it.
        if view.hand[common_guess] > 0 and self.rng.random() < 0.5:
            return PostOrder(
                common_guess, Side.SELL, Price(int(self.rng.integers(3, 9)))
            )

        return PostOrder(target, Side.BUY, Price(int(self.rng.integers(4, 10))))