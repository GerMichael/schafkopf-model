import random

from src.environment.card import Card, Suit, Rank

class Deck:
    @staticmethod
    def get_shuffled_player_cards() -> list[list[Card]]:
        cards = [Card(suit=suit, rank=rank) for suit in Suit for rank in Rank]  
        random.shuffle(cards)
        return [cards[i::4] for i in range(4)]