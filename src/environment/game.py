
from collections import defaultdict
from typing import Any, Callable

from src.environment.card import Card, Rank
from src.environment.deck import Deck
from src.environment.game_exception import GameException
from src.environment.game_modes import GameMode, GameModeType
from src.environment.game_rules import GameSpec



class Player:
    _name: str
    _hand_cards: list[Card]


    def __init__(self, name: str, hand_cards: list[Card]):
        self._name = name
        self._hand_cards = hand_cards


    @property
    def name(self):
        return self._name


    @property
    def hand_cards(self):
        return self._hand_cards
    

    def remove_hand_card(self, card: Card):
        if card not in self._hand_cards:
            raise GamePlayerHasNotCardException()
        self._hand_cards.remove(card)


class Trick:
    player_cards: list[tuple[Player, Card]]

    def __init__(self):
        self.player_cards = []

    def get_leading_card(self) -> Card | None:
        return self.player_cards[0][1] if self.player_cards else None
    
    def get_winner_player(self, game_mode: GameMode) -> Player:
        leading_card = self.get_leading_card()
        winning_card = game_mode.highest_card([c for _, c in self.player_cards], trick_suit=leading_card.suit if leading_card else None)
        return next(p for p, c in self.player_cards if c == winning_card)



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
    def __init__(self, game_mode, hand_cards: list[Card], message: str | None = None):
        if message is None:
            message = f"Non-trumpf and non-Sau card is missing for game mode {game_mode.game_mode_type.name} with suit {game_mode.suit} in hand cards: {', '.join(str(c) for c in hand_cards)}."
        super().__init__(message=message)
        self._game_mode = game_mode
        self._hand_cards = hand_cards

    @property
    def game_mode(self):
        return self._game_mode
    
    @property
    def hand_cards(self) -> list[Card]:
        return self._hand_cards
    

class GameTrickNotCompleteException(GameException):
    def __init__(self, message="Trick is not complete, yet."):
        super().__init__(message=message)



class Game:
    _players_in_order: list[Player]
    _tricks: list[Trick]
    _game_mode: GameMode
    _playing_team: list[Player]
    _defending_team: list[Player]

    def __init__(self, player_names: list[str]):
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)

        self._validate_amount_players(len(player_names))
        self._tricks = [Trick()]
        player_cards = Deck.get_shuffled_player_cards()
        self._players_in_order = [Player(name=name, hand_cards=cards) for name, cards in zip(player_names, player_cards)]
        self._playing_team = []
        self._defending_team = []

    @property
    def players_in_order(self):
        return self._players_in_order
    

    def _set_start_player(self, start_player: Player):
        new_start_player_index = self._players_in_order.index(start_player)
        self._players_in_order = self._players_in_order[new_start_player_index:] + self._players_in_order[:new_start_player_index]
    

    @property
    def played_cards(self):
        return [card for trick in self.tricks for _, card in trick.player_cards]
    
    @property
    def tricks(self):
        return self._tricks

    def get_current_trick(self):
        return self._tricks[-1]
    
    @property
    def game_mode(self):
        return self._game_mode
    
    @property
    def playing_team(self):
        return self._playing_team
    
    @property
    def defending_team(self):
        return self._defending_team
    

    def _validate_amount_players(self, amount_players: int):
        if amount_players != GameSpec.NUM_PLAYERS:
            raise GamePlayersCountException()


    def set_game_mode(self, game_mode: GameMode, playing_player: Player | None = None):
        self._validate_played_game_mode(game_mode, playing_player)
            
        self._game_mode = game_mode

        if self._game_mode.game_mode_type != GameModeType.RAMSCH:
            # Initially, in Sauspiel it is unclear who the partner is, so we assign all non-playing players to the other team.
            self._playing_team = [playing_player]
            self._defending_team = [p for p in self._players_in_order if p != playing_player]


    def _validate_played_game_mode(self, game_mode: GameMode, playing_player: Player | None):
        # Note: Not playing Ramsch implies playing either Sauspiel or something else (Solo...), but we check explicitly here to be safe for the next validation step.
        if game_mode.game_mode_type != GameModeType.RAMSCH or game_mode.game_mode_type == GameModeType.SAUSPIEL:
            if playing_player is None:
                raise GameNoPlayingPlayerException()
        
        if game_mode.game_mode_type == GameModeType.SAUSPIEL:
            any_non_trumpf_or_non_sau_of_right_suit = any(
                (
                    c.suit == game_mode.suit and 
                    c.rank != Rank.SAU and 
                    not game_mode.is_card_trumpf(c)
                ) 
                for c in playing_player.hand_cards
            )
            if not any_non_trumpf_or_non_sau_of_right_suit:
                raise GameMissingNonSauNonTrumpfCardForSauspielSuitException(game_mode=game_mode, hand_cards=playing_player.hand_cards)


    def on(self, event: str, callback: Callable[..., Any]):
        self._listeners[event].append(callback)


    def _emit(self, event: str, **kwargs: Any):
        for callback in self._listeners[event]:
            callback(**kwargs)


    def update_playing_team(self, playing_team: list[Player]):
        self._playing_team = playing_team
        self._defending_team = [p for p in self._players_in_order if p not in playing_team]


    def play_card(self, player: Player, card: Card):
        self._validate_played_card(player, card)
        player.remove_hand_card(card)
        self.get_current_trick().player_cards.append((player, card))

        if self._game_mode.game_mode_type == GameModeType.SAUSPIEL:
            is_card_searched_sau = card.rank == Rank.SAU and card.suit == self._game_mode.suit
            if is_card_searched_sau:
                partner = player
                self.update_playing_team([self._playing_team[0], partner])
                self._emit("teams_found", playing_team=self._playing_team, defending_team=self._defending_team)


    def _validate_played_card(self, player: Player, card: Card):
        num_played_cards = len(self.get_current_trick().player_cards)
        current_player = self._players_in_order[num_played_cards]
        if player != current_player:
            raise GameNotPlayersTurnException()
        if card in self.played_cards:
            raise GamePlayerAlreadyPlayedCardException()
        players_played_in_current_trick = [p for p, _ in self.get_current_trick().player_cards]
        if player in players_played_in_current_trick:
            raise GamePlayerAlreadyInTrickException()
        valid_cards = Game.get_valid_cards(player.hand_cards, self.get_current_trick().get_leading_card(), self.game_mode)
        if card not in valid_cards:
            raise GameInvalidCardException(card=card, valid_cards=valid_cards)


    def start_new_trick(self):
        len_current_trick = len(self.get_current_trick().player_cards)
        if len_current_trick != GameSpec.NUM_PLAYERS:
            raise GameTrickNotCompleteException()
        self._set_start_player(self.get_current_trick().get_winner_player(self.game_mode))
        self._tricks.append(Trick())


    def evaluate_trick(self):
        winner_card = Game.get_winning_card(self.get_current_trick(), game_mode=self._game_mode)
        winner_person = next(p for p, c in self.get_current_trick().player_cards if c == winner_card)
        return winner_person, winner_card


    @staticmethod
    def get_valid_cards(available_cards: list[Card], leading_card: Card | None, game_mode: GameMode) -> list[Card]:
        if not leading_card:
            return available_cards
        
        lead_is_trumpf = game_mode.is_card_trumpf(leading_card)
        if lead_is_trumpf:
            matching = [c for c in available_cards if game_mode.is_card_trumpf(c)]
            return matching or available_cards
        
        lead_suit = leading_card.suit
        matching = [c for c in available_cards if c.suit == lead_suit and not game_mode.is_card_trumpf(c)]
        # If the searched Sau's suit is played, the sau must be played if available
        if game_mode.game_mode_type == GameModeType.SAUSPIEL and lead_suit == game_mode.suit:
            searched_sau = Card(suit=game_mode.suit, rank=Rank.SAU)
            if searched_sau in available_cards:
                return [searched_sau]
        return matching or available_cards
    
    
    @staticmethod
    def get_winning_card(trick: Trick, game_mode: GameMode) -> Card:
        leading_card = trick.get_leading_card()
        trick_suit = leading_card.suit if leading_card else None
        return game_mode.highest_card([c for _, c in trick.player_cards], trick_suit=trick_suit) 
    