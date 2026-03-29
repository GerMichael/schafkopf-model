import pytest
from src.environment.game_modes import GameMode, GameModeType
from src.environment.card import Card, Suit, Rank


class TestIsCardTrumpf:
    def test_sauspiel_highest_herz_card(self):
        gm = GameMode(GameModeType.SAUSPIEL)
        highest_card = gm.highest_card([
            Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
            Card(suit=Suit.HERZ, rank=Rank.SAU),
            Card(suit=Suit.HERZ, rank=Rank.KOENIG),
        ])
        assert highest_card == Card(suit=Suit.HERZ, rank=Rank.SAU)

    def test_sauspiel_herz_highest_suit(self):
        gm = GameMode(GameModeType.SAUSPIEL)
        highest_card = gm.highest_card([
            Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.SAU),
            Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
        ], trick_suit=Suit.EICHEL)
        assert highest_card == Card(suit=Suit.HERZ, rank=Rank.SIEBEN)
        
    def test_sauspiel_trick_suit_among_non_trumpf(self):
        gm = GameMode(GameModeType.SAUSPIEL)
        highest_card = gm.highest_card([
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
            Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
        ], trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=Suit.GRAS, rank=Rank.SIEBEN)
    
    def test_sauspiel_ober_highest_rank(self):
        gm = GameMode(GameModeType.SAUSPIEL)
        highest_card = gm.highest_card([
            Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
            Card(suit=Suit.HERZ, rank=Rank.SAU),
            Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
        ], trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=Suit.SCHELLEN, rank=Rank.OBER)

    @pytest.mark.parametrize("cards, expected_suit", [
        ([Suit.SCHELLEN, Suit.EICHEL, Suit.HERZ, Suit.GRAS], Suit.EICHEL),
        ([Suit.SCHELLEN, Suit.HERZ, Suit.GRAS], Suit.GRAS),
        ([Suit.SCHELLEN, Suit.HERZ], Suit.HERZ),
        ([Suit.SCHELLEN], Suit.SCHELLEN),
    ])
    def test_sauspiel_highest_ober(self, cards, expected_suit):
        gm = GameMode(GameModeType.SAUSPIEL)
        ober_cards = [Card(suit=s, rank=Rank.OBER) for s in cards]
        highest_card = gm.highest_card(ober_cards, trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=expected_suit, rank=Rank.OBER)

    def test_sauspiel_unter_highest_rank(self):
        gm = GameMode(GameModeType.SAUSPIEL)
        highest_card = gm.highest_card([
            Card(suit=Suit.SCHELLEN, rank=Rank.UNTER),
            Card(suit=Suit.HERZ, rank=Rank.SAU),
            Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
        ], trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=Suit.SCHELLEN, rank=Rank.UNTER)

    @pytest.mark.parametrize("cards, expected_suit", [
        ([Suit.SCHELLEN, Suit.EICHEL, Suit.HERZ, Suit.GRAS], Suit.EICHEL),
        ([Suit.SCHELLEN, Suit.HERZ, Suit.GRAS], Suit.GRAS),
        ([Suit.SCHELLEN, Suit.HERZ], Suit.HERZ),
        ([Suit.SCHELLEN], Suit.SCHELLEN),
    ])
    def test_sauspiel_highest_unter(self, cards, expected_suit):
        gm = GameMode(GameModeType.SAUSPIEL)
        unter_cards = [Card(suit=s, rank=Rank.UNTER) for s in cards]
        highest_card = gm.highest_card(unter_cards, trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=expected_suit, rank=Rank.UNTER)

    @pytest.mark.parametrize("cards, expected_rank", [
        ([Rank.SAU, Rank.ZEHN, Rank.KOENIG, Rank.NEUN, Rank.ACHT, Rank.SIEBEN], Rank.SAU),
        ([Rank.ZEHN, Rank.KOENIG, Rank.NEUN, Rank.ACHT, Rank.SIEBEN], Rank.ZEHN),
        ([Rank.KOENIG, Rank.NEUN, Rank.ACHT, Rank.SIEBEN], Rank.KOENIG),
        ([Rank.NEUN, Rank.ACHT, Rank.SIEBEN], Rank.NEUN),
        ([Rank.ACHT, Rank.SIEBEN], Rank.ACHT),
    ])
    def test_sauspiel_highest_non_trumpf(self, cards, expected_rank):
        gm = GameMode(GameModeType.SAUSPIEL)
        non_trumpf_cards = [Card(suit=Suit.EICHEL, rank=r) for r in cards]
        highest_card = gm.highest_card(non_trumpf_cards, trick_suit=Suit.EICHEL)
        assert highest_card == Card(suit=Suit.EICHEL, rank=expected_rank)
