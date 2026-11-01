"""Console rendering and input helpers."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Sequence

from .cards import Card, Suit
from .rules import Bid, Contract, GameType, card_power, is_trump
from .scoring import RoundResult

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
COLORS = {Suit.EICHEL: "33", Suit.GRAS: "32", Suit.HERZ: "31", Suit.SCHELLEN: "36"}


class QuitGame(Exception):
    """Raised when a human types 'q' to leave the current game."""


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(text: str) -> str:
    return color(text, "1")


def card_text(card: Card, long: bool = True) -> str:
    text = str(card) if long else card.code
    return color(text, COLORS[card.suit])


def euro(cents: int) -> str:
    sign = "-" if cents < 0 else "+" if cents > 0 else " "
    return f"{sign}{abs(cents) / 100:.2f} €"


def sort_hand(hand: Sequence[Card], contract: Contract | None) -> list[Card]:
    """Sort trumps first (strongest to weakest), then the plain suits."""
    ref = contract or Contract(GameType.SAUSPIEL, 0, Suit.EICHEL)

    def key(card: Card) -> tuple[int, int, int]:
        if is_trump(card, ref):
            return (0, -card_power(card, ref, "T"), 0)
        return (1, int(card.suit), -card_power(card, ref, card.suit))

    return sorted(hand, key=key)


def clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def ask(prompt: str, input_fn: Callable[[str], str] = input) -> str:
    try:
        answer = input_fn(prompt).strip()
    except EOFError:
        raise QuitGame from None
    if answer.lower() in {"q", "quit", "exit"}:
        raise QuitGame
    return answer


def ask_choice(prompt: str, options: Sequence[str], default: int | None = None,
               input_fn: Callable[[str], str] = input) -> int:
    """Print a numbered menu and return the 0-based index of the choice."""
    for i, option in enumerate(options, 1):
        marker = " (default)" if default is not None and i - 1 == default else ""
        print(f"  [{i}] {option}{marker}")
    while True:
        answer = ask(f"{prompt} ", input_fn)
        if not answer and default is not None:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"  Please enter a number between 1 and {len(options)} (or 'q' to quit).")


def ask_int(prompt: str, default: int, low: int = 1, high: int = 10_000,
            input_fn: Callable[[str], str] = input) -> int:
    while True:
        answer = ask(f"{prompt} [{default}]: ", input_fn)
        if not answer:
            return default
        if answer.isdigit() and low <= int(answer) <= high:
            return int(answer)
        print(f"  Please enter a number between {low} and {high}.")


class ConsoleObserver:
    """Prints the public course of the game to the terminal."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.players: Sequence = ()

    def name(self, seat: int) -> str:
        return self.players[seat].name if self.players else f"Seat {seat + 1}"

    def _pause(self) -> None:
        if self.delay:
            time.sleep(self.delay)

    def round_started(self, number: int, dealer: int, players: Sequence) -> None:
        self.players = players
        print()
        print(bold(f"═══ Round {number} ═══  dealer: {self.name(dealer)}"))

    def bid_made(self, seat: int, bid: Bid) -> None:
        print(f"  {self.name(seat):>12}: {bid.label}")
        self._pause()

    def contract_set(self, contract: Contract | None) -> None:
        if contract is None:
            print(bold("  Everybody passed – the cards are thrown in."))
        elif contract.game_type is GameType.RAMSCH:
            print(bold("  Everybody passed – we play Ramsch! The most points lose."))
        else:
            print(bold(f"  {self.name(contract.declarer)} plays {contract.label}."))
        print()

    def card_played(self, seat: int, card: Card) -> None:
        print(f"  {self.name(seat):>12} plays {card_text(card)}")
        self._pause()

    def trick_completed(self, trick: Sequence[tuple[int, Card]], winner: int, points: int) -> None:
        print(f"  → {bold(self.name(winner))} takes the trick ({points} points)")
        print()
        self._pause()

    def round_finished(self, result: RoundResult | None, totals: Sequence[int]) -> None:
        if result is None:
            return
        print(bold("  " + result.summary))
        for seat in range(4):
            print(f"  {self.name(seat):>12}: {result.points[seat]:>3} points, {result.tricks[seat]} tricks,"
                  f" {euro(result.payouts[seat])}   total {euro(totals[seat])}")
