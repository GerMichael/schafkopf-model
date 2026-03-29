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

GAME_MODE_TYPES_WITH_SUIT = {GameModeType.SAUSPIEL}
GAME_MODE_TYPES_WITH_OPTIONAL_SUIT = {GameModeType.SOLO}
SAUSPIEL_VALID_SUITS = {Suit.EICHEL, Suit.GRAS, Suit.SCHELLEN}

class GameMode:
    game_mode_type: GameModeType
    suit: Suit | None = None

    def __init__(self, game_mode_type: GameModeType, suit: Suit | None = None):
        self.game_mode_type = game_mode_type
        if game_mode_type in GAME_MODE_TYPES_WITH_SUIT:
            if suit is None:
                raise ValueError(f"Suit must be specified for {game_mode_type.name} mode.")
            if game_mode_type == GameModeType.SAUSPIEL and suit not in SAUSPIEL_VALID_SUITS:
                raise ValueError(f"Suit {suit.name} is not valid for Sauspiel (Herz is always trumpf).")
            self.suit = suit
        elif game_mode_type in GAME_MODE_TYPES_WITH_OPTIONAL_SUIT:
            self.suit = suit
        else:
            if suit is not None:
                raise ValueError(f"Suit must not be specified for {game_mode_type.name} mode.")
        
    def is_card_trumpf(self, card: Card) -> bool:
        trumpf_rank_order, _, trumpf_suit_order = self.get_rank_suit_order()
        return card.rank in trumpf_rank_order or card.suit in trumpf_suit_order

    @staticmethod
    def get_suits(game_mode_type: GameModeType, hand_cards: list[Card]) -> list[Suit]:
        if game_mode_type == GameModeType.SAUSPIEL:
            suits = []
            for suit in SAUSPIEL_VALID_SUITS:
                has_sau = any(c.suit == suit and c.rank == Rank.SAU for c in hand_cards)
                has_non_sau = any(
                    (
                        c.suit == suit 
                        and c.rank != Rank.SAU 
                        and not GameMode.is_card_trumpf(GameMode(game_mode_type, Suit.EICHEL), c)
                    )
                    for c in hand_cards
                )
                if not has_sau and has_non_sau:
                    suits.append(suit)
            return sorted(suits, key=lambda s: s.value)
        return list(Suit)
    
    def get_rank_suit_order(self, trick_suit: Suit | None = None) -> tuple[list[Rank], list[Rank], list[Suit]]:
        match self.game_mode_type:
            case GameModeType.SAUSPIEL | GameModeType.RAMSCH:
                trumpf_rank_order = [Rank.OBER, Rank.UNTER]
                rest_rank_order = [Rank.SAU, Rank.ZEHN, Rank.KOENIG, Rank.NEUN, Rank.ACHT, Rank.SIEBEN]
                trumpf_suit_order = [Suit.HERZ]
            case GameModeType.SOLO:
                trumpf_rank_order = [Rank.OBER, Rank.UNTER]
                rest_rank_order = [Rank.SAU, Rank.ZEHN, Rank.KOENIG, Rank.NEUN, Rank.ACHT, Rank.SIEBEN]
                trumpf_suit_order = [self.suit] if self.suit is not None else []
            case GameModeType.WENZ:
                trumpf_rank_order = [Rank.UNTER]
                rest_rank_order = [Rank.SAU, Rank.ZEHN, Rank.KOENIG, Rank.OBER, Rank.NEUN, Rank.ACHT, Rank.SIEBEN]
                trumpf_suit_order = [self.suit] if self.suit is not None else []
            case GameModeType.GEIER:
                trumpf_rank_order = [Rank.OBER]
                rest_rank_order = [Rank.SAU, Rank.ZEHN, Rank.KOENIG, Rank.UNTER, Rank.NEUN, Rank.ACHT, Rank.SIEBEN]
                trumpf_suit_order = [self.suit] if self.suit is not None else []
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
