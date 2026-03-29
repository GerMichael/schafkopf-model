# schafkopf-model

Reinforcement learning opponent for the Bavarian card game **Schafkopf**.

## Schafkopf Game Mechanics

The underlaying Schafkopf rules are not yet configurable. Currently, the following version is implemented:

### Available Game Modes

- **Ramsch**: no extra rules, no teams, the player with the least points wins
- **Sauspiel**: no extra rules, two players play in a team, a non-Herz Sau gets searched
- **Solo**: one player against the rest, a suit can be defined as Trumpf
- **Geier**: one player against the rest, no suit is Trumpf, only Geier are Trumpf
- **Wenz**: one player against the rest, no suit is Trumpf, only Unter are Trumpf

The order of value is like this: Ramsch < Sauspiel < Solo < Geier < Wenz

### Implemented Schafkopf Procedure

1. **Setup** — 4 players are configured (via config file or manual input). A 32-card deck is shuffled and dealt evenly (8 cards per player).
2. **Bidding** — Each player views their hand and decides whether they want to play. Players who want to play choose a game mode (Sauspiel, Wenz, Geier, or Solo). The highest-valued mode wins. If nobody wants to play, Ramsch is played.
3. **Suit Selection** — The winning bidder chooses a suit if required: mandatory for Sauspiel (the searched Sau's suit), optional for Solo (a Trumpf suit).
4. **Playing Rounds** — 8 rounds are played (32 cards / 4 players). Each round:
   - A new trick starts.
   - Players play one card each in order, starting with the previous trick's winner.
   - Card validity is enforced: players must follow suit or play Trumpf if able. In Sauspiel, if the searched Sau's suit is led, the holder must play it.
   - The highest card wins the trick. The winner collects it and leads the next round.
5. **Team Reveal (Sauspiel only)** — When the searched Sau is played, teams are revealed: the Sau holder partners with the bidder.
6. **Scoring** — Each player's won trick points are summed. In Ramsch, the player with the fewest points wins. In all other modes, the player with the most points wins.

## Setup

This python projects is set up via [astral's `uv`](https://docs.astral.sh/uv/) package manager.

After installation of `uv`, execute the following to install and build the dependencies

```sh
uv sync
```

## CLI Game

Run the CLI game by executing

```
uv run cli_game.py
```

You can configure a game session config file in `config/game_session.yaml` that contains an array of four player names for the property `players`, e.g.,

```yaml
players:
 - Player A
 - Player B
 - Player C
 - Player D
```

## Testing

This project contains some test files. `pytest` handles their execution and is being configured in `pyproject.toml`.

Run all test files by executing

```sh
uv run --extra dev pytest
```

Run a specific test file by executing

```sh
uv run --extra dev pytest src/environment/game_modes_spec.py
```