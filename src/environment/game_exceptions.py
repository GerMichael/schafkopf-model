from src.environment.card import Card
from src.environment.game_modes import GameMode


class GameException(Exception):
    def __init__(self, message: str | None = None):
        super().__init__()
        self._message = message

    @property
    def message(self) -> str | None:
        return self._message
    

class GamePlayersCountException(GameException):
    def __init__(self, message="The number of players must be 4."):
        super().__init__(message=message)


class GameNoPlayingPlayerException(GameException):
    def __init__(self, message="No player chose to play a non-Ramsch game mode."):
        super().__init__(message=message)


class GameNotPlayersTurnException(GameException):
    def __init__(self, message="Not that player's turn."):
        super().__init__(message=message)


class GamePlayerAlreadyPlayedCardException(GameException):
    def __init__(self, message="Player has already played that card."):
        super().__init__(message=message)


class GamePlayerAlreadyInTrickException(GameException):
    def __init__(self, message="Player has already played a card in the current trick."):
        super().__init__(message=message)


class GamePlayerHasNotCardException(GameException):
    def __init__(self, message="Player does not have that card in hand."):
        super().__init__(message=message)


class GameInvalidCardException(GameException):
    def __init__(self, card: Card, valid_cards: list[Card], message: str | None = None):
        if message is None:
            valid_cards_str = ", ".join(str(c) for c in valid_cards)
            message = f"Invalid card played: {card}. Valid cards are: {valid_cards_str}."
        super().__init__(message=message)
        self._card = card
        self._valid_cards = valid_cards

    @property
    def card(self) -> Card:
        return self._card
    
    @property
    def valid_cards(self) -> list[Card]:
        return self._valid_cards
    

class GameMissingNonSauNonTrumpfCardForSauspielSuitException(GameException):
    def __init__(self, game_mode: GameMode, hand_cards: list[Card], message: str | None = None):
        if message is None:
            message = f"Non-trumpf and non-Sau card is missing for game mode {game_mode.game_mode_type.name} with suit {game_mode.suit} in hand cards: {', '.join(str(c) for c in hand_cards)}."
        super().__init__(message=message)
        self._game_mode = game_mode
        self._hand_cards = hand_cards

    @property
    def game_mode(self) -> GameMode:
        return self._game_mode
    
    @property
    def hand_cards(self) -> list[Card]:
        return self._hand_cards
    

class GameTrickNotCompleteException(GameException):
    def __init__(self, message="Trick is not complete, yet."):
        super().__init__(message=message)


class GameModeInvalidSuitException(GameException):
    def __init__(self, game_mode: GameMode, message: str | None = None):
        if message is None:
            message = f"Invalid suit {game_mode.suit} for game mode {game_mode.game_mode_type.name}."
        super().__init__(message=message)
        self._game_mode = game_mode

    @property
    def game_mode(self) -> GameMode:
        return self._game_mode