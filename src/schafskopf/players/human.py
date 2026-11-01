"""A human player sitting at the terminal."""

from __future__ import annotations

from collections.abc import Callable

from ..cards import Card
from ..game import Observation
from ..rules import BID_ACTIONS, GameType
from ..ui import ask, bold, card_text, clear_screen, sort_hand
from .base import Player

HELP = """  Commands: enter the number of a card/bid, 't' = show tricks and points,
            'h' = this help, 'q' = quit the game."""


class HumanPlayer(Player):
    is_human = True

    def __init__(self, name: str, hot_seat: bool = False, input_fn: Callable[[str], str] = input) -> None:
        super().__init__(name)
        self.hot_seat = hot_seat  # several humans share one screen
        self.input_fn = input_fn

    _on_screen: HumanPlayer | None = None  # whose hand is currently visible

    @property
    def kind(self) -> str:
        return "Human"

    def _hand_over(self) -> None:
        if self.hot_seat and HumanPlayer._on_screen is not self:
            ask(f"\n  >>> Pass the screen to {bold(self.name)} and press Enter <<< ", self.input_fn)
            clear_screen()
        HumanPlayer._on_screen = self

    def _show_hand(self, obs: Observation, legal: list[Card] | None = None) -> list[Card]:
        cards = sort_hand(obs.hand, obs.contract)
        print(f"\n  {bold(self.name)}, your hand:")
        for i, card in enumerate(cards, 1):
            mark = "" if legal is None or card in legal else "  (not allowed)"
            print(f"    [{i}] {card_text(card)}{mark}")
        return cards

    def _show_status(self, obs: Observation) -> None:
        print(f"  Trick {obs.trick_number + 1}/8. Points taken: "
              + ", ".join(f"seat {s + 1}: {p}" for s, p in enumerate(obs.points)))
        for n, trick in enumerate(obs.tricks_done, 1):
            print(f"    trick {n}: " + "  ".join(f"{s + 1}:{card_text(c, False)}" for s, c in trick))
        if obs.own_partner is not None:
            print(f"  Your partner is seat {obs.own_partner + 1}.")

    def bid(self, obs: Observation, legal: list[int]) -> int:
        self._hand_over()
        self._show_hand(obs)
        if obs.highest is not None:
            print(f"  Highest announcement so far: {obs.highest.label}")
        print("  What do you want to play?")
        for i, b in enumerate(legal, 1):
            print(f"    [{i}] {BID_ACTIONS[b].label}")
        while True:
            answer = ask("  Your choice: ", self.input_fn)
            if answer.lower() == "h":
                print(HELP)
            elif answer.isdigit() and 1 <= int(answer) <= len(legal):
                return legal[int(answer) - 1]
            else:
                print("  Please enter one of the numbers above.")

    def play(self, obs: Observation, legal: list[Card]) -> Card:
        self._hand_over()
        contract = obs.contract
        if contract.game_type is GameType.RAMSCH:
            role = "Ramsch – avoid points!"
        elif obs.seat == contract.declarer:
            role = "you are the declarer"
        elif obs.own_partner is not None:
            role = "you hold the called ace – you are the partner"
        else:
            role = "you play against the declarer"
        print(f"\n  Game: {bold(contract.label)} ({role})")
        if obs.trick:
            print("  On the table: " + "  ".join(f"seat {s + 1}: {card_text(c)}" for s, c in obs.trick))
        else:
            print("  You lead the trick.")
        cards = self._show_hand(obs, legal)
        while True:
            answer = ask("  Card to play: ", self.input_fn)
            if answer.lower() == "h":
                print(HELP)
            elif answer.lower() == "t":
                self._show_status(obs)
            elif answer.isdigit() and 1 <= int(answer) <= len(cards):
                card = cards[int(answer) - 1]
                if card in legal:
                    return card
                print(f"  {card} is not allowed here (follow suit / Rufsau rules).")
            else:
                print("  Please enter the number of a card.")
