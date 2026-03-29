
from src.environment.card import Card
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

    def __init__(self, player_names: list[str]):
        self.setup_new_game(player_names)
    
    def setup_new_game(self, player_names: list[str]):
        self.played_cards = []
        self.current_trick = Trick()
        player_cards = Deck.get_shuffled_player_cards()
        assert len(player_names) == GameSpec.NUM_PLAYERS, f"Exactly {GameSpec.NUM_PLAYERS} players are required."
        self.players = [Player(name=name, hand_cards=cards) for name, cards in zip(player_names, player_cards)]
        self.game_mode = GameMode(GameModeType.SAUSPIEL, None)  # Default game mode for now

    def set_game_mode(self, game_mode: GameMode):
        self.game_mode = game_mode

    def play_card(self, player: Player, card: Card):
        player.hand_cards.remove(card)
        self.current_trick.cards.append((player, card))
        self.played_cards.append(card)

    def clear_trick(self):
        self.current_trick = Trick()

    def evaluate_trick(self):
        winner_card = Game.get_winning_card(self.current_trick, game_mode=self.game_mode)
        winner_person = next(p for p, c in self.current_trick.cards if c == winner_card)
        winner_person.won_tricks.append(self.current_trick)
        return winner_person, winner_card

    @staticmethod
    def get_valid_cards(available_cards: list[Card], current_trick: Trick, game_mode: GameMode) -> list[Card]:
        if not current_trick.cards:
            return available_cards
        
        lead_is_trumpf = game_mode.is_card_trumpf(current_trick.get_leading_card())
        if lead_is_trumpf:
            matching = [c for c in available_cards if game_mode.is_card_trumpf(c)]
            return matching or available_cards
        
        lead_suit = current_trick.get_leading_card().suit
        matching = [c for c in available_cards if c.suit == lead_suit and not game_mode.is_card_trumpf(c)]
        return matching or available_cards
    
    
    @staticmethod
    def get_winning_card(trick: Trick, game_mode: GameMode) -> Card:
        return game_mode.highest_card([c for _, c in trick.cards]) 
    