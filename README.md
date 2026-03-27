# schafkopf-model

Reinforcement learning opponent for the Bavarian card game **Schafkopf**.

## Project Structure

```
schafkopf-model/
├── configs/
│   └── default.yaml            # Training hyperparameters
├── src/
│   ├── environment/            # Game logic
│   │   ├── card.py             # Card, Suit, Rank definitions
│   │   ├── deck.py             # 32-card deck, dealing
│   │   ├── game_modes.py       # Ramsch, Sauspiel, Wenz, Solo
│   │   ├── game_rules.py       # Trump order, legal plays, trick winner
│   │   ├── game_state.py       # Mutable round state
│   │   └── schafkopf_env.py    # Gymnasium-style step/reset interface
│   ├── agent/                  # RL agent
│   │   ├── network.py          # Policy & value networks (PyTorch)
│   │   ├── agent.py            # Actor-critic agent with action masking
│   │   └── random_agent.py     # Random baseline opponent
│   ├── training/               # Training pipeline
│   │   ├── rollout.py          # Experience collection (multi-agent)
│   │   ├── ppo.py              # Proximal Policy Optimization
│   │   └── self_play.py        # Opponent pool from past snapshots
│   └── utils/
│       └── logger.py           # CSV + console logging
├── tests/                      # pytest suite
├── checkpoints/                # Saved model weights
├── logs/                       # Training logs
├── train.py                    # Main training entry point
├── play.py                     # Play against the trained agent
└── pyproject.toml              # Project metadata & dependencies
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e ".[dev]"
```

## Train

```bash
python train.py --config configs/default.yaml
```

## Run Tests

```bash
pytest
```

## Play Against the Agent

```bash
python play.py --model checkpoints/final_model.pt
```