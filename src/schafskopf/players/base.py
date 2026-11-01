"""Common interface for every seat at the table (human or AI)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..cards import Card
from ..game import Observation
from ..scoring import RoundResult


class Player(ABC):
    is_human = False

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def bid(self, obs: Observation, legal: list[int]) -> int:
        """Return an index into ``rules.BID_ACTIONS`` taken from ``legal``."""

    @abstractmethod
    def play(self, obs: Observation, legal: list[Card]) -> Card:
        """Return one card from ``legal``."""

    def round_finished(self, obs: Observation, result: RoundResult | None) -> None:
        """Called once per seat after scoring (used by learning agents)."""

    @property
    def kind(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.kind}({self.name!r})"
