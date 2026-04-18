import random

from src.environment.card import Card, Suit, Rank

class Deck:
    @staticmethod
    def get_full_deck() -> list[Card]:
        return [Card(suit=suit, rank=rank) for suit in Suit for rank in Rank]

    @staticmethod
    def get_shuffled_player_cards() -> list[list[Card]]:
        cards = Deck.get_full_deck()
        random.shuffle(cards)
        return [cards[i::4] for i in range(4)]