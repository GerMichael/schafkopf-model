"""Game mode definitions for Schafkopf."""

from enum import IntEnum

from environment.card import Card, Suit, Rank


class GameModeType(IntEnum):
    RAMSCH = 0
    SAUSPIEL = 1
    WENZ = 2
    GEIER = 3
    SOLO = 4

GAME_MODE_TYPES_WITH_SUIT = {GameModeType.SOLO}

class GameMode:
    game_mode_type: GameModeType
    suit: Suit | None = None

    def __init__(self, game_mode_type: GameModeType, suit: Suit | None = None):
        self.game_mode_type = game_mode_type
        if game_mode_type in GAME_MODE_TYPES_WITH_SUIT:
            assert suit is not None, f"Suit must be specified for {GAME_MODE_TYPES_WITH_SUIT} mode."
            self.suit = suit
        
    def is_card_trumpf(self, card: Card) -> bool:
        if self.game_mode_type == GameModeType.SOLO:
            return card.suit == self.suit
        if card.suit == Suit.HERZ:
            return True
        return card.rank in (Rank.UNTER, Rank.OBER)

    def value(self) -> int:
        match self.game_mode_type:
            case GameModeType.RAMSCH:
                return 0
            case GameModeType.SAUSPIEL:
                return 1
            case GameModeType.SOLO:
                return 2
            case GameModeType.GEIER:
                return 3
            case GameModeType.WENZ:
                return 4

    def __lt__(self, other: GameMode) -> bool:
        return self.value() < other.value()