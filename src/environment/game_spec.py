import random

import pytest
from src.environment.game import Game
from src.environment.game_modes import GameMode, GameModeType, get_valid_game_mode_types_for_cards
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


class TestGameFlow:

    def test_random_game_sessions(self):
        for _ in range(5):
            game = Game(player_names=["Alice", "Bob", "Charlie", "Diana"])
            players = game.players_in_sitting_order
            playing_player = random.choice(players)
            valid_game_types = get_valid_game_mode_types_for_cards(playing_player.hand_cards)
            played_game_mode_type = random.choice(valid_game_types)
            valid_suits = GameMode.get_suits_for_cards(played_game_mode_type, playing_player.hand_cards)
            if len(valid_suits) == 0:
                played_suit = None
            else:
                played_suit = random.choice(valid_suits)
            game.set_game_mode(
                game_mode=GameMode(game_mode_type=played_game_mode_type, suit=played_suit),
                playing_player=playing_player,
                )

            for _ in range(8):
                print(f"Trick players: {[p.name for p in game.get_players_order_for_current_trick()]}")
                for player in game.get_players_order_for_current_trick():
                    valid_cards = Game.get_valid_cards(player.hand_cards, game.get_current_trick().get_first_card(), game.game_mode)
                    card_to_play = random.choice(valid_cards)
                    game.play_card(player, card_to_play)
                winning_player, winning_card = game.evaluate_current_trick()
                print(f"Trick winner: {winning_player.name} with {winning_card}")
                game.start_new_trick()


    def test_sauspiel_team_assignment_and_reveal(self):
        alice_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.SIEBEN),
            Card(suit=Suit.EICHEL, rank=Rank.ACHT),
            Card(suit=Suit.EICHEL, rank=Rank.NEUN),
            Card(suit=Suit.EICHEL, rank=Rank.KOENIG),
            Card(suit=Suit.GRAS, rank=Rank.SIEBEN),
            Card(suit=Suit.GRAS, rank=Rank.ACHT),
            Card(suit=Suit.GRAS, rank=Rank.NEUN),
            Card(suit=Suit.GRAS, rank=Rank.KOENIG),
        ]
        bob_cards = [
            Card(suit=Suit.SCHELLEN, rank=Rank.SIEBEN),
            Card(suit=Suit.SCHELLEN, rank=Rank.ACHT),
            Card(suit=Suit.SCHELLEN, rank=Rank.NEUN),
            Card(suit=Suit.SCHELLEN, rank=Rank.KOENIG),
            Card(suit=Suit.HERZ, rank=Rank.SIEBEN),
            Card(suit=Suit.HERZ, rank=Rank.ACHT),
            Card(suit=Suit.HERZ, rank=Rank.NEUN),
            Card(suit=Suit.HERZ, rank=Rank.KOENIG),
        ]
        charlie_cards = [
            Card(suit=Suit.EICHEL, rank=Rank.ZEHN),
            Card(suit=Suit.EICHEL, rank=Rank.SAU),
            Card(suit=Suit.EICHEL, rank=Rank.UNTER),
            Card(suit=Suit.EICHEL, rank=Rank.OBER),
            Card(suit=Suit.GRAS, rank=Rank.ZEHN),
            Card(suit=Suit.GRAS, rank=Rank.SAU),
            Card(suit=Suit.GRAS, rank=Rank.UNTER),
            Card(suit=Suit.GRAS, rank=Rank.OBER),
        ]
        diana_cards = [
            Card(suit=Suit.SCHELLEN, rank=Rank.ZEHN),
            Card(suit=Suit.SCHELLEN, rank=Rank.SAU),
            Card(suit=Suit.SCHELLEN, rank=Rank.UNTER),
            Card(suit=Suit.SCHELLEN, rank=Rank.OBER),
            Card(suit=Suit.HERZ, rank=Rank.ZEHN),
            Card(suit=Suit.HERZ, rank=Rank.SAU),
            Card(suit=Suit.HERZ, rank=Rank.UNTER),
            Card(suit=Suit.HERZ, rank=Rank.OBER),
        ]

        game = Game(
            player_names=["Alice", "Bob", "Charlie", "Diana"], 
            player_cards=[alice_cards, bob_cards, charlie_cards, diana_cards]
        )

        alice, bob, charlie, diana = game.players_in_sitting_order
        
        game.set_game_mode(
            game_mode=GameMode(GameModeType.SAUSPIEL, Suit.EICHEL), 
            playing_player=alice
        )
        
        assert game.playing_team == [alice]
        assert game.defending_team == [bob, charlie, diana]
        
        events = []
        def on_teams_found(**kwargs):
            events.append(kwargs)
            
        game.on("teams_found", on_teams_found)
        
        game.play_card(alice, Card(Suit.EICHEL, Rank.SIEBEN))
        assert len(events) == 0
        game.play_card(bob, Card(Suit.SCHELLEN, Rank.SIEBEN))
        assert len(events) == 0
        game.play_card(charlie, Card(Suit.EICHEL, Rank.SAU))
        assert len(events) == 1
        
        event_args = events[0]
        assert event_args["playing_team"] == [alice, charlie]
        assert event_args["defending_team"] == [bob, diana]
        
        assert game.playing_team == [alice, charlie]
        assert game.defending_team == [bob, diana]
