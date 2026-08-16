"""Deterministic randomness.

Every random draw in Parity comes from a `numpy.random.Generator` obtained from
this module. Nothing calls `random.random()` or `np.random.seed()` -- global RNG
state is invisible, order-dependent, and does not survive multiprocessing.

The important design decision, made on day 1 because it is painful to retrofit:

    the deal and each agent get SEPARATE, INDEPENDENT streams.

That is what makes common random numbers possible later. If the dealer and the
agents shared a stream, then changing an agent would change how many draws it
consumed, which would change the *deal* -- and the paired comparison you built
the whole tournament harness for would silently be unpaired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Generator = np.random.Generator


@dataclass(frozen=True, slots=True)
class GameStreams:
    """Independent RNG streams for a single game, derived from one master seed."""

    deal: Generator
    agents: tuple[Generator, ...]


def spawn(master_seed: int, n: int) -> list[Generator]:
    """n independent generators derived from `master_seed`.

    Uses SeedSequence spawning, which gives statistically independent streams --
    unlike `default_rng(seed + i)`, where neighbouring seeds are correlated in
    ways that will quietly bias a Monte Carlo study.
    """
    root = np.random.SeedSequence(master_seed)
    return [np.random.default_rng(child) for child in root.spawn(n)]


def game_streams(master_seed: int, game_index: int, n_agents: int) -> GameStreams:
    """Streams for game `game_index` of a tournament run under `master_seed`.

    Reproducible in isolation: game 7000 gives the same deal whether you run
    games 0..9999 or only game 7000, on one core or on sixteen.
    """
    per_game = np.random.SeedSequence([master_seed, game_index])
    deal_ss, agent_root = per_game.spawn(2)
    agent_seqs = agent_root.spawn(n_agents)
    return GameStreams(
        deal=np.random.default_rng(deal_ss),
        agents=tuple(np.random.default_rng(s) for s in agent_seqs),
    )