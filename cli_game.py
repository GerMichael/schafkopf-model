import os

import yaml

from src.environment.game_modes import GAME_MODE_TYPES_WITH_SUIT, GameMode, GameModeType
from src.environment.card import Card, Card, Suit
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


def _format_card(card, valid = True) -> str:
    label = str(card)
    if not valid:
        return f"{_DIM}{label}{_RESET}"
    color = _SUIT_COLORS[card.suit]
    return f"{_BOLD}{color}{label}{_RESET}"


class CliGame:
    def _get_game_config(self) -> dict:
        if os.path.exists("game_config.yaml"):
            with open("game_config.yaml", "r") as f:
                config = yaml.safe_load(f)
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
            sorted_cards = sorted(player.hand_cards)
            self._prompt_cards(sorted_cards, valid_cards=player.hand_cards)
            while True:
                choice = input(f"{player.name}, do you want to play? (y/n): ").strip().lower()
                if choice in ('y', 'n'):
                    player_choices.append((player, choice == 'y'))
                    break
                print("Invalid input. Please enter 'y' or 'n'.")
        return player_choices
    
    def _determine_game_mode(self, player_choices: list[tuple[Player, bool]]) -> tuple[Player | None, GameMode]:
        player_game_mode_types: list[tuple[Player, GameModeType | None]] = []
        for player, wants_to_play in player_choices:
            if not wants_to_play:
                player_game_mode_types.append((player, None))
                continue
            while True:
                mode_choice = input(f"{player.name}, choose game mode (1: Sauspiel, 2: Wenz, 3: Geier, 4: Solo): ").strip()
                if mode_choice in ('1', '2', '3', '4'):
                    player_game_mode_types.append((player, GameModeType(int(mode_choice))))
                    break
                print("Invalid input. Please enter a number between 1 and 4.")
            suit = None
        highest_game_mode_type = GameModeType.highest([gm for _, gm in player_game_mode_types])
        playing_player = next((p for p, gmt in player_game_mode_types if gmt == highest_game_mode_type), None)
        if highest_game_mode_type in GAME_MODE_TYPES_WITH_SUIT:
            while True:
                suit_choice = input(f"{playing_player.name}, choose suit for {highest_game_mode_type.name} (0: Eichel, 1: Gras, 2: Herz, 3: Schellen): ").strip()
                if suit_choice in ('0', '1', '2', '3'):
                    suit = Suit(int(suit_choice))
                    break
                print("Invalid input. Please enter a number between 0 and 3.")
            highest_game_mode = GameMode(highest_game_mode_type, suit)
        else:
            highest_game_mode = GameMode(highest_game_mode_type)
        return playing_player, highest_game_mode
    
    def _prompt_cards(self, cards: list[Card], valid_cards: list[Card]):
        print(f"Your cards:")
        for idx, card in enumerate(cards):
            is_valid = card in valid_cards
            formatted = _format_card(card, is_valid)
            marker = " " if is_valid else "x"
            print(f"  [{marker}] {idx}: {formatted}")
    
    def _prompt_card_choice(self, player: Player):
        valid_cards = set(Game.get_valid_cards(player.hand_cards, self.game.current_trick, self.game.game_mode))
        sorted_cards = sorted(player.hand_cards)
        self._prompt_cards(sorted_cards, valid_cards)
        while True:
            try:
                choice = int(input("Choose card index: "))
                if 0 <= choice < len(sorted_cards) and sorted_cards[choice] in valid_cards:
                    return sorted_cards[choice]
                print("Invalid choice. Pick a valid (unhighlighted) card.")
            except ValueError:
                print("Please enter a number.")

    def _print_results(self):
        total_winner_player = None
        total_won_points = 0
        for player in self.game.players:
            player_points = sum(c.value() for trick in player.won_tricks for _, c in trick.cards)
            print(f"{player.name} scored {player_points} points.")
            if player_points > total_won_points:
                total_won_points = player_points
                total_winner_player = player

        if total_winner_player:
            print(f"\n{total_winner_player.name} wins the game with {total_won_points} points!")

    def run(self):
        game_config = self._get_game_config()
        print(f"Using game config: {game_config}")
        player_names = self._get_player_names(game_config)
        self.game = Game(player_names)
        self.players_order = list(self.game.players)

        player_that_want_to_play = self._ask_players_to_play(self.game.players)
        playing_player, game_mode = self._determine_game_mode(player_that_want_to_play)
        self.game.set_game_mode(game_mode)
        # TODO: Find teams for Sauspiel
        if game_mode.game_mode_type == GameModeType.RAMSCH:
            print("\nNo one wants to play. Starting Ramsch.")
        else:
            print(f"\n{playing_player.name} will play {game_mode.game_mode_type.name}.")

        num_rounds = GameSpec.NUM_CARDS // GameSpec.NUM_PLAYERS
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1} ---")
            self.game.clear_trick()

            for player in self.players_order:
                print(f"\nIt's your turn, {player.name}.")
                chosen = self._prompt_card_choice(player)
                self.game.play_card(player, chosen)
                print(f"  -> {player.name} plays {_format_card(chosen)}")

            winner_person, winner_card = self.game.evaluate_trick()
            won_points = sum(c.value() for _, c in self.game.current_trick.cards)
            print(f"\n{winner_person.name} wins the trick with {_format_card(winner_card)} ({won_points} points)!")
            winner_idx = self.players_order.index(winner_person)
            self.players_order = self.players_order[winner_idx:] + self.players_order[:winner_idx]

        print("\nGame over!")
        self._print_results()


if __name__ == "__main__":
    CliGame().run()
