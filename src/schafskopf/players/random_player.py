"""A player that picks uniformly among the legal options."""

from __future__ import annotations

import random

from ..cards import Card
from ..game import Observation
from ..rules import PASS
from .base import Player


class RandomPlayer(Player):
    def __init__(self, name: str, rng: random.Random | None = None, bid_rate: float = 0.15) -> None:
        super().__init__(name)
        self.rng = rng or random.Random()
        self.bid_rate = bid_rate

    @property
    def kind(self) -> str:
        return "Random"

    def bid(self, obs: Observation, legal: list[int]) -> int:
        options = [b for b in legal if b != PASS]
        if options and self.rng.random() < self.bid_rate:
            return self.rng.choice(options)
        return PASS

    def play(self, obs: Observation, legal: list[Card]) -> Card:
        return self.rng.choice(legal)
