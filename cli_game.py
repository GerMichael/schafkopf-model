import os

import yaml

from src.environment.game_modes import GAME_MODE_TYPES_WITH_SUIT, GAME_MODE_TYPES_WITH_OPTIONAL_SUIT, GameMode, GameModeType, get_highest_game_mode_type
from src.environment.card import Card, Suit
from src.environment.game import Game, Player
from src.environment.game_rules import GameSpec

# ANSI color codes per suit
_SUIT_COLORS = {
    Suit.EICHEL: "\033[34m",    # Blue
    Suit.GRAS: "\033[32m",      # Green
    Suit.HERZ: "\033[31m",      # Red
    Suit.SCHELLEN: "\033[33m",  # Yellow
}
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _format_suit(suit: Suit) -> str:
    color = _SUIT_COLORS[suit]
    return f"{color}{suit.name}{_RESET}"


def _format_card(card, valid = True) -> str:
    label = str(card)
    if not valid:
        return f"{_DIM}{label}{_RESET}"
    color = _SUIT_COLORS[card.suit]
    return f"{_BOLD}{color}{label}{_RESET}"


def _sort_cards_by_suit_and_rank_value(cards: list[Card]) -> list[Card]:
    return sorted(cards, key=lambda c: (c.suit.value, c.rank.value))


class CliGame:
    _game: Game | None = None


    def _get_game_session_config(self) -> dict:
        config_path = os.path.join(os.getcwd(), "config", "game_session.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if config is not None:
                return config
        return {}
    

    def _get_player_names(self, game_config: dict) -> list[str]:
        player_names = game_config.get("players", [])
        print(f"Found {len(player_names)} player names in config: {player_names}")
        if len(player_names) != GameSpec.NUM_PLAYERS:
            print(f"Config file must contain exactly {GameSpec.NUM_PLAYERS} player names under 'players'. Falling back to manual input.")
            player_names = self._ask_for_player_names()
        else:
            return player_names


    def _ask_for_player_names(self) -> list[str]:
        player_names = []
        for i in range(GameSpec.NUM_PLAYERS):
            name = input(f"Enter player {i + 1} name: ")
            player_names.append(name)
        return player_names
    

    def _ask_players_to_play(self, players: list[Player]) -> list[tuple[Player, bool]]:
        player_choices = []
        for player in players:
            sorted_cards = _sort_cards_by_suit_and_rank_value(player.hand_cards)
            self._prompt_cards(sorted_cards, valid_cards=player.hand_cards)
            while True:
                choice = input(f"{player.name}, do you want to play? (y/n): ").strip().lower()
                if choice in ('y', 'n'):
                    player_choices.append((player, choice == 'y'))
                    break
                print("Invalid input. Please enter 'y' or 'n'.")
        return player_choices
    

    def _ask_game_mode_type(self, player: Player) -> GameModeType:
        playable_modes = sorted((gmt for gmt in GameModeType if gmt != GameModeType.RAMSCH), key=lambda g: g.value)
        sauspiel_valid_suits = GameMode.get_suits(GameModeType.SAUSPIEL, player.hand_cards)
        playable_modes = [gmt for gmt in playable_modes if gmt != GameModeType.SAUSPIEL or sauspiel_valid_suits]
        options = ", ".join(f"{gmt.value}: {gmt.name}" for gmt in playable_modes)
        valid = {str(gmt.value) for gmt in playable_modes}
        while True:
            choice = input(f"{player.name}, choose game mode ({options}): ").strip()
            if choice in valid:
                return GameModeType(int(choice))
            print(f"Invalid input. Please enter one of: {', '.join(sorted(valid))}.")


    def _ask_suit(self, player: Player, game_mode_type: GameModeType) -> Suit | None:
        optional = game_mode_type in GAME_MODE_TYPES_WITH_OPTIONAL_SUIT
        allowed_suits = GameMode.get_suits(game_mode_type, player.hand_cards)
        suit_options = ", ".join(f"{s.value}: {s.name}" for s in allowed_suits)
        valid = {str(s.value) for s in allowed_suits}
        if optional:
            suit_options += ", N: None"
        while True:
            choice = input(f"{player.name}, choose suit for {game_mode_type.name} ({suit_options}): ").strip()
            if choice in valid:
                return Suit(int(choice))
            if optional and choice.upper() == 'N':
                return None
            print("Invalid input.")


    def _determine_game_mode(self, player_choices: list[tuple[Player, bool]]) -> tuple[Player | None, GameMode]:
        player_game_mode_types: list[tuple[Player, GameModeType | None]] = []
        for player, wants_to_play in player_choices:
            if not wants_to_play:
                player_game_mode_types.append((player, None))
            else:
                player_game_mode_types.append((player, self._ask_game_mode_type(player)))

        highest_game_mode_type = get_highest_game_mode_type([gm for _, gm in player_game_mode_types])
        playing_player = next((p for p, gmt in player_game_mode_types if gmt == highest_game_mode_type), None)

        suit = None
        if highest_game_mode_type in GAME_MODE_TYPES_WITH_SUIT | GAME_MODE_TYPES_WITH_OPTIONAL_SUIT:
            suit = self._ask_suit(playing_player, highest_game_mode_type)

        return playing_player, GameMode(highest_game_mode_type, suit)
    

    def _prompt_cards(self, cards: list[Card], valid_cards: list[Card]):
        print("Your cards:")
        for idx, card in enumerate(cards):
            is_valid = card in valid_cards
            formatted = _format_card(card, is_valid)
            marker = " " if is_valid else "x"
            print(f"  [{marker}] {idx}: {formatted}")
    

    def _prompt_card_choice_to_play(self, player: Player):
        valid_cards = Game.get_valid_cards(player.hand_cards, self._game.get_current_trick().get_leading_card(), self._game.game_mode)
        trumpf_cards = [c for c in player.hand_cards if self._game.game_mode.is_card_trumpf(c)]
        non_trumpf_cards = [c for c in player.hand_cards if not self._game.game_mode.is_card_trumpf(c)]
        sorted_non_trumpf_cards = _sort_cards_by_suit_and_rank_value(non_trumpf_cards)
        sorted_trumpf_cards = _sort_cards_by_suit_and_rank_value(trumpf_cards)
        sorted_cards = sorted_non_trumpf_cards + sorted_trumpf_cards
        self._prompt_cards(sorted_cards, valid_cards)
        while True:
            try:
                choice = int(input("Choose card index: "))
                if 0 <= choice < len(sorted_cards) and sorted_cards[choice] in valid_cards:
                    return sorted_cards[choice]
                print("Invalid choice. Pick one of the valid (highlighted) cards without an 'x' marker.")
            except ValueError:
                print("Please enter a number.")


    def _on_teams_found(self, playing_team: list[Player], defending_team: list[Player]):
        playing_names = ", ".join(p.name for p in playing_team)
        defending_names = ", ".join(p.name for p in defending_team)
        print(f"\nTeams revealed! Playing: {playing_names} | Defending: {defending_names}")


    def _print_results(self):
        total_winner_player = None
        total_won_points = 0
        for player in self._game.players_in_order:
            player_points = sum(c.value() for trick in self._game.tricks for p, c in trick.player_cards if trick.get_winner_player(self._game.game_mode) == player)
            print(f"{player.name} scored {player_points} points.")
            if self._game.game_mode.game_mode_type != GameModeType.RAMSCH:
                if total_winner_player is None or player_points > total_won_points:
                    total_won_points = player_points
                    total_winner_player = player
            else: 
                if total_winner_player is None or player_points < total_won_points:
                    total_won_points = player_points
                    total_winner_player = player

        if total_winner_player:
            print(f"\n{total_winner_player.name} wins the game with {total_won_points} points!")


    def run(self):
        game_config = self._get_game_session_config()
        print(f"Using game config: {game_config}")
        player_names = self._get_player_names(game_config)
        self._game = Game(player_names)
        self._game.on("teams_found", self._on_teams_found)

        player_that_want_to_play = self._ask_players_to_play(self._game.players_in_order)
        playing_player, game_mode = self._determine_game_mode(player_that_want_to_play)
        self._game.set_game_mode(game_mode=game_mode, playing_player=playing_player)
        
        if game_mode.game_mode_type == GameModeType.RAMSCH:
            print("\nNo one wants to play. Starting Ramsch.")
        elif game_mode.game_mode_type in GAME_MODE_TYPES_WITH_OPTIONAL_SUIT and game_mode.suit is None:
            print(f"\n{playing_player.name} will play {game_mode.game_mode_type.name} (no suit).")
        elif game_mode.suit is not None:
            print(f"\n{playing_player.name} will play {_format_suit(game_mode.suit)} {game_mode.game_mode_type.name}.")
        else:
            print(f"\n{playing_player.name} will play {game_mode.game_mode_type.name}.")

        num_rounds = GameSpec.NUM_CARDS // GameSpec.NUM_PLAYERS
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1} ---")

            for player in self._game.players_in_order:
                print(f"\nIt's your turn, {player.name}.")
                chosen = self._prompt_card_choice_to_play(player)
                self._game.play_card(player, chosen)
                print(f"  -> {player.name} plays {_format_card(chosen)}")

            winner_person, winner_card = self._game.evaluate_trick()
            won_points = sum(c.value() for _, c in self._game.get_current_trick().player_cards)
            print(f"\n{winner_person.name} wins the trick with {_format_card(winner_card)} ({won_points} points)!")
            
            self._game.start_new_trick()

        print("\nGame over!")
        self._print_results()


if __name__ == "__main__":
    CliGame().run()
