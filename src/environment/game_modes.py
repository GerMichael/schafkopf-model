"""Game mode definitions for Schafkopf."""

from enum import IntEnum

from src.environment.card import Card, Suit, Rank


class GameModeType(IntEnum):
    RAMSCH = 0
    SAUSPIEL = 1
    SOLO = 2
    GEIER = 3
    WENZ = 4

    @staticmethod
    def highest(game_mode_types: list[GameModeType | None]) -> GameModeType:
        choices = [gmt for gmt in game_mode_types if gmt is not None]
        if not choices:
            return GameModeType.RAMSCH
        return max(choices)

GAME_MODE_TYPES_WITH_SUIT = {GameModeType.SOLO, GameModeType.SAUSPIEL}

class GameMode:
    game_mode_type: GameModeType
    suit: Suit | None = None

    def __init__(self, game_mode_type: GameModeType, suit: Suit | None = None):
        self.game_mode_type = game_mode_type
        if game_mode_type in GAME_MODE_TYPES_WITH_SUIT:
            assert suit is not None, f"Suit must be specified for {GAME_MODE_TYPES_WITH_SUIT} mode."
            self.suit = suit
        
    def is_card_trumpf(self, card: Card) -> bool:
        trumpf_rank_order, _, trumpf_suit_order = self.get_rank_suit_order()
        return card.rank in trumpf_rank_order or card.suit in trumpf_suit_order
    
    def get_rank_suit_order(self, trick_suit: Suit | None = None) -> tuple[list[Rank], list[Rank], list[Suit]]:
        match self.game_mode_type:
            case GameModeType.SAUSPIEL | GameModeType.RAMSCH:
                trumpf_rank_order = [Rank.OBER, Rank.UNTER]
                rest_rank_order = [Rank.SAU, Rank.ZEHN, Rank.KOENIG, Rank.NEUN, Rank.ACHT, Rank.SIEBEN]
                trumpf_suit_order = [Suit.HERZ]
            case GameModeType.SOLO:
                trumpf_rank_order = [Rank.OBER, Rank.UNTER]
                rest_rank_order = [Rank.SAU, Rank.ZEHN, Rank.KOENIG, Rank.NEUN, Rank.ACHT, Rank.SIEBEN]
                trumpf_suit_order = [self.suit]
            case GameModeType.WENZ:
                trumpf_rank_order = [Rank.UNTER]
                rest_rank_order = [Rank.SAU, Rank.ZEHN, Rank.KOENIG, Rank.OBER, Rank.NEUN, Rank.ACHT, Rank.SIEBEN]
                trumpf_suit_order = [self.suit] if self.suit else []
            case GameModeType.GEIER:
                trumpf_rank_order = [Rank.OBER]
                rest_rank_order = [Rank.SAU, Rank.ZEHN, Rank.KOENIG, Rank.UNTER, Rank.NEUN, Rank.ACHT, Rank.SIEBEN]
                trumpf_suit_order = [self.suit] if self.suit else []
        if trick_suit and trick_suit not in trumpf_suit_order:
            trumpf_suit_order.append(trick_suit)
        return trumpf_rank_order, rest_rank_order, trumpf_suit_order
    
    def highest_card(self, cards: list[Card], trick_suit: Suit | None = None) -> Card:
        trumpf_rank_order, rest_rank_order, trumpf_suit_order = self.get_rank_suit_order(trick_suit)
        def card_sort_key(card: Card):
            if card.rank in trumpf_rank_order:
                return (0, trumpf_rank_order.index(card.rank), card.suit)
            elif card.suit in trumpf_suit_order:
                return (1, trumpf_suit_order.index(card.suit), rest_rank_order.index(card.rank))
            else:
                return (2, rest_rank_order.index(card.rank))
        return min(cards, key=card_sort_key)

    def __lt__(self, other: GameMode) -> bool:
        return self.game_mode_type < other.game_mode_type
