
from collections import defaultdict
from typing import Any, Callable

from src.environment.card import Card, Rank, Suit
from src.environment.deck import Deck
from src.environment.game_modes import GameMode, GameModeType
from src.environment.game_rules import GameSpec


class Player:
    name: str
    hand_cards: list[Card]
    won_tricks: list[Trick]

    def __init__(self, name: str, hand_cards: list[Card]):
        self.name = name
        self.hand_cards = hand_cards
        self.won_tricks = []


class Trick:
    cards: list[tuple[Player, Card]]

    def __init__(self):
        self.cards = []

    def get_leading_card(self) -> Card | None:
        return self.cards[0][1] if self.cards else None


class Game:
    players: list[Player]
    played_cards: list[Card]
    current_trick: Trick
    game_mode: GameMode
    playing_team: list[Player]
    defending_team: list[Player]

    def __init__(self, player_names: list[str]):
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        self.setup_new_game(player_names)
    
    def setup_new_game(self, player_names: list[str]):
        self.played_cards = []
        self.current_trick = Trick()
        player_cards = Deck.get_shuffled_player_cards()
        assert len(player_names) == GameSpec.NUM_PLAYERS, f"Exactly {GameSpec.NUM_PLAYERS} players are required."
        self.players = [Player(name=name, hand_cards=cards) for name, cards in zip(player_names, player_cards)]
        self.playing_team = []
        self.defending_team = []

    def set_game_mode(self, game_mode: GameMode, playing_player: Player | None = None):
        self.game_mode = game_mode
        if self.game_mode.game_mode_type != GameModeType.RAMSCH:
            if playing_player is None:
                raise ValueError("playing_player is required for non-RAMSCH game modes")
            # Initially, in Sauspiel it is unclear who the partner is, so we assign all non-playing players to the other team.
            self.playing_team = [playing_player]
            self.defending_team = [p for p in self.players if p != playing_player]

    def on(self, event: str, callback: Callable[..., Any]):
        self._listeners[event].append(callback)

    def _emit(self, event: str, **kwargs: Any):
        for callback in self._listeners[event]:
            callback(**kwargs)

    def update_playing_team(self, playing_team: list[Player]):
        self.playing_team = playing_team
        self.defending_team = [p for p in self.players if p not in playing_team]

    def play_card(self, player: Player, card: Card):
        player.hand_cards.remove(card)
        self.current_trick.cards.append((player, card))
        self.played_cards.append(card)

        if self.game_mode.game_mode_type == GameModeType.SAUSPIEL:
            is_card_searched_sau = card.rank == Rank.SAU and card.suit == self.game_mode.suit
            if is_card_searched_sau:
                partner = player
                self.update_playing_team([self.playing_team[0], partner])
                self._emit("teams_found", playing_team=self.playing_team, defending_team=self.defending_team)

    def clear_trick(self):
        self.current_trick = Trick()

    def evaluate_trick(self):
        winner_card = Game.get_winning_card(self.current_trick, game_mode=self.game_mode)
        winner_person = next(p for p, c in self.current_trick.cards if c == winner_card)
        winner_person.won_tricks.append(self.current_trick)
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
        return game_mode.highest_card([c for _, c in trick.cards], trick_suit=trick_suit) 
    