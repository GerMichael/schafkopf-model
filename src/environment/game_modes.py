"""Game mode definitions for Schafkopf."""

from enum import IntEnum


class GameMode(IntEnum):
    RAMSCH = 0
    SAUSPIEL = 1
    WENZ = 2
    SOLO = 3
