import pytest
from src.environment.game import Game
from src.environment.game_modes import GameMode, GameModeType
from src.environment.card import Card, Suit, Rank


class TestGameValidCards:
    
    @pytest.mark.parametrize("available_cards, leading_card, valid_cards", [
        ( # Trumpf cards available -> only trumpf cards playable
            [
                Card(suit=Suit.EICHEL, rank=Rank.SAU),
                Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
                Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            ], 
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
            [
                Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
                Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            ]
        ),
        ( # No trumpf hand cards -> all playable
            [
                Card(suit=Suit.EICHEL, rank=Rank.SAU),
                Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
            ], 
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
            [
                Card(suit=Suit.EICHEL, rank=Rank.SAU),
                Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
            ]
        ),
    ])
    def test_sauspiel_leading_trumpf(self, available_cards, leading_card, valid_cards):
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == set(valid_cards)
        
    @pytest.mark.parametrize("available_cards, leading_card, valid_cards", [
        ( # Leading suit is among non-trumpf cards
            [
                Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
                Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
                Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            ], 
            Card(suit=Suit.EICHEL, rank=Rank.KOENIG),
            [
                Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            ]
        ),
        ( # Leading suit is not among non-trumpf cards -> all playable
            [
                Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
                Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            ], 
            Card(suit=Suit.EICHEL, rank=Rank.KOENIG),
            [
                Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
                Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            ]
        ),
        ( # Only trumpf hand cards -> all playable
            [
                Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
                Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            ], 
            Card(suit=Suit.EICHEL, rank=Rank.KOENIG),
            [
                Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
                Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            ]
        ),
    ])
    def test_sauspiel_leading_color(self, available_cards, leading_card, valid_cards):
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == set(valid_cards)
    
    def test_sauspiel_sau_searched(self):
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
        available_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
            Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
        ]
        leading_card = Card(suit=Suit.EICHEL, rank=Rank.KOENIG)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == {
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
        }
    
    def test_sauspiel_sau_on_hand_and_leading(self):
        gm = GameMode(GameModeType.SAUSPIEL, Suit.EICHEL)
        available_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
            Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
        ]
        leading_card = None
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == {
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
            Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
        }

    def test_solo_no_suit_trumpf(self):
        gm = GameMode(GameModeType.SOLO, None)
        available_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
            Card(suit=Suit.EICHEL, rank=Rank.OBER),
        ]
        leading_card = Card(suit=Suit.EICHEL, rank=Rank.UNTER)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == {
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
            Card(suit=Suit.EICHEL, rank=Rank.OBER),
        }
    
    def test_solo_no_suit_no_trumpf(self):
        gm = GameMode(GameModeType.SOLO, None)
        available_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
        ]
        leading_card = Card(suit=Suit.EICHEL, rank=Rank.UNTER)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == {
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
        }

    @pytest.mark.parametrize("available_cards, leading_card, valid_cards", [
        ( # Trumpf cards available -> only trumpf cards playable
            [
                Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
                Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
                Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
                Card(suit=Suit.HERZ, rank=Rank.SAU),
            ], 
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
            [
                Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
                Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            ]
        ),
        ( # Trumpf played and heart available -> heart not playable
            [
                Card(suit=Suit.GRAS, rank=Rank.SAU),
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
                Card(suit=Suit.HERZ, rank=Rank.ACHT),
            ], 
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            [
                Card(suit=Suit.GRAS, rank=Rank.UNTER),
            ]
        ),
    ])
    def test_solo_suit_trumpf(self, available_cards, leading_card, valid_cards):
        gm = GameMode(GameModeType.SOLO, Suit.EICHEL)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == set(valid_cards)
    
    @pytest.mark.parametrize("available_cards, leading_card, valid_cards", [
        ( # No Trumpf cards except Heart available, Trumpf played -> all cards playable
            [
                Card(suit=Suit.GRAS, rank=Rank.SAU),
                Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
                Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
            ], 
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
            [
                Card(suit=Suit.GRAS, rank=Rank.SAU),
                Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
                Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
            ]
        ),
    ])
    def test_solo_suit_no_trumpf(self, available_cards, leading_card, valid_cards):
        gm = GameMode(GameModeType.SOLO, Suit.EICHEL)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == set(valid_cards)

    def test_wenz_trumpf(self):
        gm = GameMode(GameModeType.WENZ, None)
        available_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
            Card(suit=Suit.EICHEL, rank=Rank.OBER),
        ]
        leading_card = Card(suit=Suit.EICHEL, rank=Rank.UNTER)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == {
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
        }
    
    def test_wenz_no_trumpf(self):
        gm = GameMode(GameModeType.WENZ, None)
        available_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.EICHEL, rank=Rank.OBER),
        ]
        leading_card = Card(suit=Suit.EICHEL, rank=Rank.UNTER)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == {
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.EICHEL, rank=Rank.OBER),
        }
    
    def test_geier_trumpf(self):
        gm = GameMode(GameModeType.GEIER, None)
        available_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.OBER),
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
        ]
        leading_card = Card(suit=Suit.EICHEL, rank=Rank.OBER)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == {
            Card(suit=Suit.GRAS, rank=Rank.OBER),
        }

    def test_geier_no_trumpf(self):
        gm = GameMode(GameModeType.GEIER, None)
        available_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
        ]
        leading_card = Card(suit=Suit.EICHEL, rank=Rank.OBER)
        actual_valid_cards = Game.get_valid_cards(available_cards, leading_card, game_mode=gm)
        assert set(actual_valid_cards) == {
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
        }
