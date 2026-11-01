"""Cards of the Bavarian (German-suited) 32-card Schafkopf deck."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    """The four suits, in their traditional ranking order."""

    EICHEL = 0
    GRAS = 1
    HERZ = 2
    SCHELLEN = 3

    @property
    def label(self) -> str:
        return self.name.capitalize()

    @property
    def symbol(self) -> str:
        return SUIT_SYMBOLS[self]

    @property
    def short(self) -> str:
        return self.name[0]


class Rank(IntEnum):
    """Card ranks. The integer value is only an index, not a strength."""

    ASS = 0
    ZEHN = 1
    KOENIG = 2
    OBER = 3
    UNTER = 4
    NEUN = 5
    ACHT = 6
    SIEBEN = 7

    @property
    def label(self) -> str:
        return RANK_LABELS[self]

    @property
    def short(self) -> str:
        return RANK_SHORT[self]


SUIT_SYMBOLS = {Suit.EICHEL: "♣", Suit.GRAS: "♠", Suit.HERZ: "♥", Suit.SCHELLEN: "♦"}
RANK_LABELS = {
    Rank.ASS: "Sau",
    Rank.ZEHN: "Zehn",
    Rank.KOENIG: "König",
    Rank.OBER: "Ober",
    Rank.UNTER: "Unter",
    Rank.NEUN: "Neun",
    Rank.ACHT: "Acht",
    Rank.SIEBEN: "Sieben",
}
RANK_SHORT = {
    Rank.ASS: "A",
    Rank.ZEHN: "10",
    Rank.KOENIG: "K",
    Rank.OBER: "O",
    Rank.UNTER: "U",
    Rank.NEUN: "9",
    Rank.ACHT: "8",
    Rank.SIEBEN: "7",
}
CARD_POINTS = {
    Rank.ASS: 11,
    Rank.ZEHN: 10,
    Rank.KOENIG: 4,
    Rank.OBER: 3,
    Rank.UNTER: 2,
    Rank.NEUN: 0,
    Rank.ACHT: 0,
    Rank.SIEBEN: 0,
}


@dataclass(frozen=True, order=True, slots=True)
class Card:
    suit: Suit
    rank: Rank

    @property
    def index(self) -> int:
        """Stable position 0..31 used for vector encodings."""
        return int(self.suit) * 8 + int(self.rank)

    @property
    def points(self) -> int:
        return CARD_POINTS[self.rank]

    @classmethod
    def from_index(cls, index: int) -> Card:
        return ALL_CARDS[index]

    @property
    def name(self) -> str:
        return f"{self.suit.label}-{self.rank.label}"

    @property
    def code(self) -> str:
        return f"{self.suit.symbol}{self.rank.short}"

    def __str__(self) -> str:
        return f"{self.code} {self.name}"


ALL_CARDS: tuple[Card, ...] = tuple(Card(s, r) for s in Suit for r in Rank)
NUM_CARDS = len(ALL_CARDS)


def new_deck(rng: random.Random | None = None) -> list[Card]:
    deck = list(ALL_CARDS)
    (rng or random).shuffle(deck)
    return deck


def deal(rng: random.Random | None = None, players: int = 4) -> list[list[Card]]:
    """Deal 8 cards to each of four players (traditionally 4 + 4)."""
    deck = new_deck(rng)
    size = len(deck) // players
    return [deck[i * size : (i + 1) * size] for i in range(players)]
