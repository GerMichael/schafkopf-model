import pytest
from src.environment.game_modes import GameMode, GameModeInvalidSuitException, GameModeType
from src.environment.card import Card, Suit, Rank


class TestGameModeValidation:
    def test_sauspiel_rejects_herz(self):
        with pytest.raises(GameModeInvalidSuitException):
            GameMode(GameModeType.SAUSPIEL, Suit.HERZ)

    @pytest.mark.parametrize("suit", [Suit.EICHEL, Suit.GRAS, Suit.SCHELLEN])
    def test_sauspiel_accepts_non_herz(self, suit):
        gm = GameMode(GameModeType.SAUSPIEL, suit)
        assert gm.suit == suit

    def test_sauspiel_suits_with_non_sau_card(self):
        hand = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.KOENIG),
            Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
            Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
        ]
        assert GameMode.get_suits(GameModeType.SAUSPIEL, hand) == [Suit.GRAS]

    def test_sauspiel_suits_only_sau_no_other(self):
        hand = [
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
        ]
        assert GameMode.get_suits(GameModeType.SAUSPIEL, hand) == []


class TestIsCardTrumpf:
    def test_sauspiel_highest_herz_card(self):
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
        highest_card = gm.highest_card([
            Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
            Card(suit=Suit.HERZ, rank=Rank.SAU),
            Card(suit=Suit.HERZ, rank=Rank.KOENIG),
        ])
        assert highest_card == Card(suit=Suit.HERZ, rank=Rank.SAU)

    def test_sauspiel_herz_highest_suit(self):
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
        highest_card = gm.highest_card([
            Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.SAU),
            Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
        ], trick_suit=Suit.EICHEL)
        assert highest_card == Card(suit=Suit.HERZ, rank=Rank.SIEBEN)
        
    def test_sauspiel_trick_suit_among_non_trumpf(self):
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
        highest_card = gm.highest_card([
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
            Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
        ], trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=Suit.GRAS, rank=Rank.SIEBEN)
    
    def test_sauspiel_ober_highest_rank(self):
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
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
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
        ober_cards = [Card(suit=s, rank=Rank.OBER) for s in cards]
        highest_card = gm.highest_card(ober_cards, trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=expected_suit, rank=Rank.OBER)

    def test_sauspiel_unter_highest_rank(self):
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
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
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
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
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
        non_trumpf_cards = [Card(suit=Suit.EICHEL, rank=r) for r in cards]
        highest_card = gm.highest_card(non_trumpf_cards, trick_suit=Suit.EICHEL)
        assert highest_card == Card(suit=Suit.EICHEL, rank=expected_rank)

    def test_solo_called_suit(self):
        gm = GameMode(GameModeType.SOLO, Suit.GRAS)
        highest_card = gm.highest_card([
            Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
            Card(suit=Suit.HERZ, rank=Rank.SAU),
        ], trick_suit=Suit.HERZ)
        assert highest_card == Card(suit=Suit.GRAS, rank=Rank.SIEBEN)

    def test_solo_no_called_suit(self):
        gm = GameMode(GameModeType.SOLO, None)
        highest_card = gm.highest_card([
            Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
            Card(suit=Suit.HERZ, rank=Rank.SAU),
        ], trick_suit=Suit.HERZ)
        assert highest_card == Card(suit=Suit.HERZ, rank=Rank.SAU)

    def test_solo_highest_ober(self):
        gm = GameMode(GameModeType.SOLO, Suit.GRAS)
        highest_card = gm.highest_card([
            Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            Card(suit=Suit.GRAS, rank=Rank.OBER),
            Card(suit=Suit.EICHEL, rank=Rank.OBER),
            Card(suit=Suit.HERZ, rank=Rank.OBER),
        ], trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=Suit.EICHEL, rank=Rank.OBER)
    
    def test_solo_highest_unter(self):
        gm = GameMode(GameModeType.SOLO, Suit.GRAS)
        highest_card = gm.highest_card([
            Card(suit=Suit.SCHELLEN, rank=Rank.UNTER),
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
            Card(suit=Suit.HERZ, rank=Rank.UNTER),
        ], trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=Suit.EICHEL, rank=Rank.UNTER)

    def test_geier_ober_higher_than_unter(self):
        gm = GameMode(GameModeType.GEIER)
        highest_card = gm.highest_card([
            Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
            Card(suit=Suit.HERZ, rank=Rank.UNTER),
        ], trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=Suit.SCHELLEN, rank=Rank.OBER)
    
    def test_geier_unter_is_non_rank(self):
        gm = GameMode(GameModeType.GEIER)
        highest_card = gm.highest_card([
            Card(suit=Suit.SCHELLEN, rank=Rank.UNTER),
            Card(suit=Suit.SCHELLEN, rank=Rank.KOENIG),
        ], trick_suit=Suit.SCHELLEN)
        assert highest_card == Card(suit=Suit.SCHELLEN, rank=Rank.KOENIG)

    def test_wenz_unter_higher_than_ober(self):
        gm = GameMode(GameModeType.WENZ)
        highest_card = gm.highest_card([
            Card(suit=Suit.SCHELLEN, rank=Rank.UNTER),
            Card(suit=Suit.GRAS, rank=Rank.OBER),
            Card(suit=Suit.EICHEL, rank=Rank.OBER),
            Card(suit=Suit.HERZ, rank=Rank.OBER),
        ], trick_suit=Suit.GRAS)
        assert highest_card == Card(suit=Suit.SCHELLEN, rank=Rank.UNTER)
    
    def test_wenz_ober_is_non_rank(self):
        gm = GameMode(GameModeType.WENZ)
        highest_card = gm.highest_card([
            Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            Card(suit=Suit.SCHELLEN, rank=Rank.KOENIG),
        ], trick_suit=Suit.SCHELLEN)
        assert highest_card == Card(suit=Suit.SCHELLEN, rank=Rank.KOENIG)
