"""Game mode definitions for Schafkopf."""

from enum import IntEnum

from src.environment.card import Card, Suit, Rank
from src.environment.game_exception import GameException


class GameModeType(IntEnum):
    RAMSCH = 0
    SAUSPIEL = 1
    SOLO = 2
    GEIER = 3
    WENZ = 4


GAME_MODE_TYPES_WITH_SUIT = {GameModeType.SAUSPIEL}
GAME_MODE_TYPES_WITH_OPTIONAL_SUIT = {GameModeType.SOLO}
SAUSPIEL_VALID_SUITS = {Suit.EICHEL, Suit.GRAS, Suit.SCHELLEN}


def get_highest_game_mode_type(game_mode_types: list[GameModeType | None]) -> GameModeType:
    choices = [gmt for gmt in game_mode_types if gmt is not None]
    if not choices:
        return GameModeType.RAMSCH
    return max(choices)


class GameModeInvalidSuitException(GameException):
    def __init__(self, game_mode, message: str | None = None):
        if message is None:
            message = f"Invalid suit {game_mode.suit} for game mode {game_mode.game_mode_type.name}."
        super().__init__(message=message)
        self._game_mode = game_mode

    @property
    def game_mode(self):
        return self._game_mode
    


class GameMode:
    _game_mode_type: GameModeType
    _suit: Suit | None = None


    def __init__(self, game_mode_type: GameModeType, suit: Suit | None = None):
        self._game_mode_type = game_mode_type
        if game_mode_type in GAME_MODE_TYPES_WITH_SUIT:
            if suit is None:
                raise GameModeInvalidSuitException(game_mode=self, message=f"Suit must be specified for {game_mode_type.name} mode.")
            if game_mode_type == GameModeType.SAUSPIEL and suit not in SAUSPIEL_VALID_SUITS:
                raise GameModeInvalidSuitException(game_mode=self, message=f"Suit {suit.name} is not valid for Sauspiel (Herz is always trumpf).")
            self._suit = suit
        elif game_mode_type in GAME_MODE_TYPES_WITH_OPTIONAL_SUIT:
            self._suit = suit
        else:
            if suit is not None:
                raise GameModeInvalidSuitException(game_mode=self, message=f"Suit must not be specified for {game_mode_type.name} mode.")
    

    @property
    def game_mode_type(self) -> GameModeType:
        return self._game_mode_type
    

    @property
    def suit(self) -> Suit | None:
        return self._suit


    def is_card_trumpf(self, card: Card) -> bool:
        trumpf_rank_order, _, trumpf_suit_order = self.get_rank_suit_order()
        return card.rank in trumpf_rank_order or card.suit in trumpf_suit_order
    

    def has_non_sau_and_non_trumpf_cards_for_suit(self, cards: list[Card]) -> bool:
        """
        Returns a boolean indicating whether the game mode has any cards of its suit that are not the Sau and not trumpf.
        This method is useful to check if a list of cards contains a non-trumpf non-Sau card of the game mode's suit, so a Sauspiel of that suit can be played.
        """
        return any(
                    (
                        c.suit == self.suit
                        and c.rank != Rank.SAU 
                        and not self.is_card_trumpf(c)
                    )
                    for c in cards
                )


    @staticmethod
    def get_suits(game_mode_type: GameModeType, hand_cards: list[Card]) -> list[Suit]:
        if game_mode_type == GameModeType.SAUSPIEL:
            suits = []
            for suit in SAUSPIEL_VALID_SUITS:
                sauspiel_of_suit = GameMode(GameModeType.SAUSPIEL, suit)
                has_sau = any(c.suit == suit and c.rank == Rank.SAU for c in hand_cards)
                has_non_sau = sauspiel_of_suit.has_non_sau_and_non_trumpf_cards_for_suit(hand_cards)
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


    def __lt__(self, other: object) -> bool:
        if not isinstance(other, GameMode):
            raise NotImplementedError()
        return self.game_mode_type < other.game_mode_type

