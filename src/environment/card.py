"""Card, Suit, and Rank definitions for the 32-card Schafkopf deck."""

from enum import IntEnum


class Suit(IntEnum):
    EICHEL = 0
    GRAS = 1
    HERZ = 2
    SCHELLEN = 3


class Rank(IntEnum):
    SIEBEN = 0
    ACHT = 1
    NEUN = 2
    UNTER = 3
    OBER = 4
    KOENIG = 5
    ZEHN = 6
    SAU = 7
