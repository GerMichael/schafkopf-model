# Schafkopf

A complete text-based version of **Schafkopf**, the Bavarian trick-taking card
game, written in Python 3.14 and managed with [uv](https://docs.astral.sh/uv/).
Play against humans (hot-seat) or AI opponents in three difficulty tiers
(easy, medium, hard). The AI is trained with **reinforcement learning** on
state vectors that are stored in a separate folder and can be retrained by hand
at any time.

## Quick start

```bash
uv sync                 # install Python 3.14 environment + numpy + PyTorch
uv run schafskopf       # interactive menu: start a game, pick mode and players
```

Train or improve the AI opponents:

```bash
uv run schafskopf train-all                         # all tiers with config/tiers.toml
uv run schafskopf train --tier hard --episodes 20000 # continue training one tier
uv run schafskopf evaluate --tier hard              # how good is it?
uv run schafskopf data                              # models + stored vectors
```

## Features

* All four players, full rules: following suit and trump, Rufsau obligations
  (incl. *davonlaufen*), Sauspiel, Wenz, Farbsolo, Ramsch (with Jungfrau and
  Durchmarsch), Schneider, Schwarz, Laufende and a configurable tariff.
* Table modes: Standard, without Ramsch, Sauspiel only, Ramsch only, Custom.
* Seats can be Human, AI easy/medium/hard, a rule-based bot or a random bot.
  Several humans share the screen in hot-seat mode; with no humans you can watch.
* AI = two PyTorch neural networks per tier (bidding + card play), warm-started by
  imitating the rule-based bot and then trained by self-play reinforcement
  learning (actor-critic for card play, contextual bandit for bidding).
* Training vectors in `training_data/<tier>/`, model files in `models/`,
  hyper-parameters in `config/tiers.toml` – all editable by hand.

## Documentation

| Document | Content |
|----------|---------|
| [docs/gameplay.md](docs/gameplay.md) | installing, menus, playing a round step by step, commands |
| [docs/rules.md](docs/rules.md) | deck, card values, trumps, following suit, Rufsau, bidding |
| [docs/game-modes.md](docs/game-modes.md) | Sauspiel, Wenz, Solo, Ramsch and the table modes |
| [docs/winning-and-scoring.md](docs/winning-and-scoring.md) | 61 points, Schneider, Schwarz, Laufende, payouts, match winner |
| [docs/ai-and-training.md](docs/ai-and-training.md) | tiers, RL algorithm, vector layout, training data, retraining by hand |
| [docs/architecture.md](docs/architecture.md) | code structure, engine flow, extending, tests |

## Development

```bash
uv run pytest
```

Rules reference: https://www.sauspiel.de/schafkopf-lernen
