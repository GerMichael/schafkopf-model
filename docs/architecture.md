# Architecture

Schafkopf is a text-based Schafkopf game with a rule engine, rule-based and
random bots, and reinforcement-learning (RL) opponents built with PyTorch.
The engine is independent of the AI; the AI only talks to the engine through
the `Player` interface.

Related documents: [how-to-extend.md](how-to-extend.md),
[known-issues.md](known-issues.md), [ai-and-training.md](ai-and-training.md),
[rules.md](rules.md).

## 1. Repository layout

```
schafkopf-model/
├── pyproject.toml           uv project (Python ≥ 3.14, numpy, torch)
├── config/tiers.toml        hand-editable AI tier settings
├── models/                  trained networks: easy.npz, medium.npz, hard.npz
├── training_data/           training vectors, one folder per tier
│   └── <tier>/batch_*.npz, imitation_*.npz, log.csv
├── docs/                    documentation
├── tests/                   pytest suite (test_rules.py, test_ai.py)
├── .github/workflows/       CI: `uv sync --locked`, `uv run pytest` on every push
└── src/schafskopf/
    ├── cards.py             Suit, Rank, Card, deck, dealing
    ├── rules.py             GameType, Bid, Contract, trumps, legal moves, trick winner
    ├── scoring.py           Tariff, winning conditions, payouts (RoundResult)
    ├── game.py              Observation, Round (one deal), Match (series), GameSettings
    ├── ui.py                console rendering, input helpers, ConsoleObserver
    ├── cli.py               menus and the `schafskopf` command
    ├── players/
    │   ├── base.py          Player interface
    │   ├── human.py         terminal player (hot-seat aware)
    │   ├── heuristic.py     rule-based bot (+ hand_strength)
    │   └── random_player.py random legal moves
    └── ai/
        ├── config.py        paths (SCHAFSKOPF_HOME) and TierConfig loader
        ├── encoding.py      Observation → state vectors (bid / play)
        ├── network.py       PolicyValueNet: PyTorch MLP, policy + value heads, Adam
        ├── agent.py         ModelBundle (model file) and RLPlayer
        └── training.py      Trainer: imitation, self-play, replay, evaluate
```

## 2. Layers and dependencies

```
cli.py ─► game.py ─► rules.py ─► cards.py
  │          │            └────► scoring.py
  │          └──── Player (players/base.py)
  ├─► players/{human,heuristic,random_player}
  └─► ai/agent.py ─► ai/network.py (torch)
        ai/training.py ─► agent, encoding, config, network
```

* `cards`, `rules`, `scoring`, `game` form the **engine**. They have no
  dependency on numpy, torch or the console.
* `players` implement the `Player` interface. `ai.agent.RLPlayer` is just
  another `Player`.
* `ai` is the only package that imports torch. The `cli` imports it lazily
  (`make_ai`), so playing without AI tiers does not load PyTorch.

## 3. Flow of one round

```
Match.play_round()
 └─ Round.run()
     ├─ run_bidding()   for each seat from dealer+1:  player.bid(observation, legal_bids)
     ├─ run_play()      8 tricks × 4 cards:           player.play(observation, legal_cards)
     ├─ score_round()   → RoundResult (payouts per seat, sum is zero)
     └─ player.round_finished(observation, result)    (learning agents label their moves)
```

* The engine never trusts players: every returned bid/card is checked against
  the legal list.
* Players only see an `Observation`: their own hand plus public information
  (bids, tricks, points) and private knowledge (e.g. the Rufsau holder knows
  the partnership). Hidden hands are never passed to a player.
* Observers (`ConsoleObserver`) receive public events for display. The engine
  itself prints nothing, which keeps training fast.

## 4. The AI

Each tier (`easy`, `medium`, `hard`) owns a **`ModelBundle`** stored as
`models/<tier>.npz`, holding two `PolicyValueNet` instances:

| Network | Input | Output | Learning |
|---------|-------|--------|----------|
| `bid_net`  | `BID_DIM` vector (`encode_bid`)   | Q(s, a) for each announcement | contextual-bandit regression (`q_step`) |
| `play_net` | `PLAY_DIM` vector (`encode_play`) | 32 card logits + V(s)         | actor-critic policy gradient (`train_step`) |

`PolicyValueNet` (`ai/network.py`) is a `torch.nn.Module`:

```
x ─ Linear ─ ReLU ─ … ─┬─ policy_head → logits (masked softmax over legal moves)
                       └─ value_head  → V(s)
```

Design points:

* **numpy at the boundary.** `predict`, `logits`, `train_step` and `q_step`
  take and return numpy arrays; torch is an implementation detail of the net.
  Agents, encoding and the trainer are unaware of it.
* **Masking.** Illegal moves get a logit of `-1e9` before the softmax, so the
  network can never choose them.
* **Seat-relative encoding.** All seat-dependent features are expressed
  relative to the observing seat, so one network plays any seat.
* **Optimizer lives in the net.** Each net owns its `torch.optim.Adam`; the
  learning rate is set per call from `TierConfig.learning_rate`. Gradient
  norms are clipped (`max_grad_norm`).
* **Persistence.** `ModelBundle.save` writes weights *and* Adam moments into
  one compressed `.npz` using a stable array layout (weights stored as
  `(in, out)` matrices, names `bid.W0`, `bid.adam_m.W0`, …). This is the same
  layout the earlier numpy implementation used, so older model files load.
  The file is written to a temporary name and renamed, so an interrupted
  save does not corrupt a model.

### Training pipeline (`ai/training.py`)

```
fresh model ─► imitate()   copy the rule-based bot (supervised, warm start)
            └► train()     loop: collect() self-play batch
                                 → trajectories_to_arrays()
                                 → save_batch()   (training_data/<tier>/batch_*.npz)
                                 → learn()        (update_net on both nets)
                                 → checkpoint model, append log.csv
replay()    re-learn from stored batches
evaluate()  play N rounds against a baseline and report payout per round
```

Configuration comes from `config/tiers.toml` (`TierConfig`). Project paths
resolve via `SCHAFSKOPF_HOME`, then the nearest parent directory containing
`config/tiers.toml`, then the source tree.

## 5. Command line

`schafskopf` (entry point `schafskopf.cli:main`) offers an interactive menu
and sub-commands for training (`train`), replaying stored data (`replay`),
inspecting data and evaluating a tier against a baseline. Player kinds are
listed in `cli.PLAYER_TYPES` and instantiated by `cli.make_player`.

## 6. Tests and CI

```bash
uv run pytest
```

* `tests/test_rules.py` – card points, trump orders, follow-suit, all Rufsau
  rules, bidding legality, Laufende, payouts, full random rounds (zero-sum,
  120 points).
* `tests/test_ai.py` – encoding sizes and legality of recorded actions,
  masking and learning behaviour of `train_step`/`q_step`, model save/load
  round trip, a mini end-to-end training run, tier config parsing.

GitHub Actions (`.github/workflows/run_pytests.yaml`) runs the suite on every
push using the locked dependencies (`uv sync --locked`).
