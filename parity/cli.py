"""Command line. The first part of this project you actually *run*.

    python -m parity.cli --seed 42
    python -m parity.cli --seed 42 --agents partner,hand,random,random
    python -m parity.cli --games 500 --agents partner,hand,partner,hand
"""

from __future__ import annotations

import argparse
from collections import Counter

from parity.agents import REGISTRY, build
from parity.dealer import deal
from parity.events import OrderRejected, Passed, Traded
from parity.loop import DEFAULT_EVENT_BUDGET, play
from parity.rng import game_streams


def _play_one(names: list[str], seed: int, game: int, budget: int) -> None:
    streams = game_streams(seed, game, n_agents=len(names))
    d = deal(streams.deal, n_players=len(names))
    agents = [build(n) for n in names]
    result = play(d, agents, event_budget=budget, streams=streams)

    c = d.composition
    print(f"\ndeck        : {c}")
    print(f"common suit : {c.common_suit.name} (12 cards, worth nothing)")
    print(f"goal suit   : {c.goal_suit.name} ({c.goal_suit_size} cards)")
    print(f"bonus       : {c.bonus} to the leader, majority at {c.majority_threshold}")

    trades = [e for e in result.log if isinstance(e, Traded)]
    rejects = [e for e in result.log if isinstance(e, OrderRejected)]
    passes = [e for e in result.log if isinstance(e, Passed)]

    print(f"\n{len(trades)} trades over {result.n_turns} turns "
          f"({len(passes)} passes, {len(rejects)} rejected)")

    if trades:
        print("\nfirst 10 trades")
        for e in trades[:10]:
            t = e.trade
            print(f"    {t.suit.name[:1]} @{t.price:>2}   "
                  f"seat{t.seller} -> seat{t.buyer}   ({t.aggressor.value} lifted)")

    print("\n                    S  C  H  D   goal   payout    pnl")
    for i, name in enumerate(names):
        start, end = d.hands[i], result.final_hands[i]
        print(f"  seat {i} {name:<10} {end[0]:>2} {end[1]:>2} {end[2]:>2} {end[3]:>2}"
              f"   {end[c.goal_suit]:>4}   {result.payouts[i]:>6}   {result.pnl[i]:>+5}"
              f"    (dealt {start[c.goal_suit]} goal cards)")

    print(f"\n  payouts sum to {sum(result.payouts)}   "
          f"pnl sums to {sum(result.pnl):+d}   winner: seat {result.winner()}")


def _play_many(names: list[str], seed: int, games: int, budget: int) -> None:
    totals = [0] * len(names)
    wins: Counter[int] = Counter()

    for g in range(games):
        streams = game_streams(seed, g, n_agents=len(names))
        d = deal(streams.deal, n_players=len(names))
        agents = [build(n) for n in names]
        result = play(d, agents, event_budget=budget, streams=streams)
        for i, p in enumerate(result.pnl):
            totals[i] += p
        wins[int(result.winner())] += 1

    print(f"\n{games} games, seed {seed}, {budget} turns each\n")
    print("  seat  agent        mean pnl    wins")
    for i, name in enumerate(names):
        print(f"  {i:>4}  {name:<10} {totals[i] / games:>+9.2f}  {wins[i]:>6}")
    print(f"\n  mean pnl sums to {sum(totals) / games:+.2f}  (must be 0.00)")
    print("\n  NOTE: seats are not rotated, so this is not yet a fair comparison.")
    print("  Same agent in different seats sees different flow. The tournament")
    print("  harness fixes that with a Latin square and reports intervals.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="parity", description="Play Figgie.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--agents",
        default="partner,hand,random,random",
        help=f"comma-separated, one per seat. available: {','.join(sorted(REGISTRY))}",
    )
    parser.add_argument("--turns", type=int, default=DEFAULT_EVENT_BUDGET)
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()

    names = [n.strip() for n in args.agents.split(",")]
    for name in names:
        if name not in REGISTRY:
            parser.error(f"unknown agent {name!r}; have {sorted(REGISTRY)}")
    if len(names) not in (4, 5):
        parser.error(f"need 4 or 5 agents, got {len(names)}")

    if args.games == 1:
        _play_one(names, args.seed, game=0, budget=args.turns)
    else:
        _play_many(names, args.seed, games=args.games, budget=args.turns)


if __name__ == "__main__":
    main()