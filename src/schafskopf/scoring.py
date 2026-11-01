"""Winning conditions and payouts (Tarif) for a finished round."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .cards import Card
from .rules import Contract, GameType, trump_order

WIN_THRESHOLD = 61  # the declaring side needs at least 61 of 120 points
SCHNEIDER_THRESHOLD = 30  # a side with 30 points or less is "Schneider"


@dataclass(frozen=True, slots=True)
class Tariff:
    """Payout amounts in cents. Edit to play with your local tariff."""

    sauspiel: int = 10
    solo: int = 50  # also used for Wenz
    ramsch: int = 10
    schneider: int = 10
    schwarz: int = 10
    laufende: int = 10
    min_laufende: int = 3  # Sauspiel and Solo
    min_laufende_wenz: int = 2


@dataclass(slots=True)
class RoundResult:
    contract: Contract | None
    payouts: list[int]  # per seat, in cents, sums to zero
    points: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    tricks: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    declarer_team: tuple[int, ...] = ()
    declarer_points: int = 0
    declarers_won: bool = False
    schneider: bool = False
    schwarz: bool = False
    laufende: int = 0
    value: int = 0
    ramsch_losers: tuple[int, ...] = ()
    durchmarsch: int | None = None
    summary: str = ""


def count_laufende(contract: Contract, hands: Sequence[Sequence[Card]], team: set[int]) -> int:
    """Consecutive top trumps held by one side, counted from the highest trump."""
    owner = {card: seat for seat, hand in enumerate(hands) for card in hand}
    order = trump_order(contract)
    first_side = owner[order[0]] in team
    count = 0
    for card in order:
        if (owner[card] in team) != first_side:
            break
        count += 1
    return count


def score_round(
    contract: Contract,
    initial_hands: Sequence[Sequence[Card]],
    points: Sequence[int],
    tricks: Sequence[int],
    partner: int | None,
    tariff: Tariff = Tariff(),
) -> RoundResult:
    if contract.game_type is GameType.RAMSCH:
        return _score_ramsch(contract, points, tricks, tariff)

    declarer = contract.declarer
    team = {declarer} if partner is None else {declarer, partner}
    decl_points = sum(points[s] for s in team)
    decl_tricks = sum(tricks[s] for s in team)
    won = decl_points >= WIN_THRESHOLD

    loser_points = 120 - decl_points if won else decl_points
    loser_tricks = (8 - decl_tricks) if won else decl_tricks
    schneider = loser_points <= SCHNEIDER_THRESHOLD
    schwarz = loser_tricks == 0

    solo_like = contract.game_type in (GameType.SOLO, GameType.WENZ)
    value = tariff.solo if solo_like else tariff.sauspiel
    if schneider:
        value += tariff.schneider
    if schwarz:
        value += tariff.schwarz
    laufende = count_laufende(contract, initial_hands, team)
    minimum = tariff.min_laufende_wenz if contract.game_type is GameType.WENZ else tariff.min_laufende
    if laufende >= minimum:
        value += laufende * tariff.laufende
    else:
        laufende = 0

    sign = 1 if won else -1
    payouts = [0, 0, 0, 0]
    opponents = [s for s in range(4) if s not in team]
    if len(team) == 1:
        payouts[declarer] = sign * value * 3
        for s in opponents:
            payouts[s] = -sign * value
    else:
        for s in team:
            payouts[s] = sign * value
        for s in opponents:
            payouts[s] = -sign * value

    extras = [label for flag, label in ((schneider, "Schneider"), (schwarz, "Schwarz")) if flag]
    if laufende:
        extras.append(f"{laufende} Laufende")
    side = "Declarers win" if won else "Declarers lose"
    summary = f"{contract.label}: {side} with {decl_points} points"
    if extras:
        summary += " (" + ", ".join(extras) + ")"
    summary += f" – worth {value} ct per player"

    return RoundResult(
        contract=contract,
        payouts=payouts,
        points=list(points),
        tricks=list(tricks),
        declarer_team=tuple(sorted(team)),
        declarer_points=decl_points,
        declarers_won=won,
        schneider=schneider,
        schwarz=schwarz,
        laufende=laufende,
        value=value,
        summary=summary,
    )


def _score_ramsch(contract: Contract, points: Sequence[int], tricks: Sequence[int], tariff: Tariff) -> RoundResult:
    """Ramsch: everybody plays alone, the player with the most points loses.

    A player who takes all eight tricks wins a Durchmarsch instead. Players
    with no trick at all ("Jungfrau") double the loser's payment to them.
    """
    payouts = [0, 0, 0, 0]
    sweeper = next((s for s in range(4) if tricks[s] == 8), None)
    if sweeper is not None:
        value = tariff.ramsch * 3
        for s in range(4):
            payouts[s] = value * 3 if s == sweeper else -value
        return RoundResult(
            contract, payouts, list(points), list(tricks), durchmarsch=sweeper, value=value,
            summary=f"Ramsch: Durchmarsch by seat {sweeper + 1} – worth {value} ct per player",
        )

    top = max(points)
    losers = tuple(s for s in range(4) if points[s] == top)
    winners = [s for s in range(4) if s not in losers]
    for w in winners:
        amount = tariff.ramsch * (2 if tricks[w] == 0 else 1)
        payouts[w] += amount * len(losers)
        for l in losers:
            payouts[l] -= amount
    names = ", ".join(f"seat {s + 1}" for s in losers)
    return RoundResult(
        contract, payouts, list(points), list(tricks), ramsch_losers=losers, value=tariff.ramsch,
        summary=f"Ramsch: {names} lose{'s' if len(losers) == 1 else ''} with {top} points",
    )
