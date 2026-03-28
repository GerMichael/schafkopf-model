"""Card, Suit, and Rank definitions for the 32-card Schafkopf deck."""

from enum import IntEnum


class Suit(IntEnum):
    EICHEL = 0
    GRAS = 1
    HERZ = 2
    SCHELLEN = 3


class Rank(IntEnum):
    SIEBEN = 1
    ACHT = 2
    NEUN = 3
    UNTER = 4
    OBER = 5
    KOENIG = 6
    ZEHN = 7
    SAU = 8

class Card:
    suit: Suit
    rank: Rank

    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    def value(self) -> int:
        match(self.rank):
            case Rank.SIEBEN:
                return 0
            case Rank.ACHT:
                return 0
            case Rank.NEUN:
                return 0
            case Rank.UNTER:
                return 2
            case Rank.OBER:
                return 3
            case Rank.KOENIG:
                return 4
            case Rank.ZEHN:
                return 10
            case Rank.SAU:
                return 11

    def __str__(self):
        return f"{self.suit.name} {self.rank.name}"

    def __lt__(self, other: "Card") -> bool:
        if self.suit != other.suit:
            return self.suit < other.suit
        return self.rank < other.rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self) -> int:
        return hash((self.suit, self.rank))
