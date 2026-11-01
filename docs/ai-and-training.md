# AI Opponents and Reinforcement-Learning Training

## 1. Overview

Every AI tier (**easy**, **medium**, **hard**) consists of two small neural
networks, implemented from scratch in numpy (`src/schafskopf/ai/network.py`):

| Network       | Input                     | Output                                   | Learns with |
|---------------|---------------------------|------------------------------------------|-------------|
| **bid net**   | 76-dim bidding vector     | 9 values Q(s, a): expected reward of each announcement | contextual-bandit regression |
| **play net**  | 231-dim card-play vector  | 32 logits → policy π(card \| s) + value V(s) | actor-critic policy gradient |

Both share the architecture `input → Dense+ReLU → Dense+ReLU → heads`; the hidden
sizes are set per tier. Illegal actions are always masked out, so the AI can
never break a rule.

The tiers differ in three ways (all set in `config/tiers.toml`):

| Tier   | Network      | Imitation warm start | Self-play RL              | At the table |
|--------|--------------|----------------------|---------------------------|--------------|
| easy   | `[64, 32]`   | 2 000 rounds         | 5 000 rounds vs random bots | samples moves (temperature 1.0) + 10 % random moves |
| medium | `[128, 64]`  | 8 000 rounds         | 20 000 rounds, mixed table  | low temperature 0.3 + 2 % random moves |
| hard   | `[256, 128]` | 30 000 rounds        | 60 000 rounds vs rule-based bots | always its best move (temperature 0) |

So the tiers differ in network size, amount and kind of training, and in how
deterministically they play. All of it is set in `config/tiers.toml`.

Files:

```
config/tiers.toml               hyper-parameters (edit by hand)
models/<tier>.npz               network weights + Adam state + metadata
training_data/<tier>/
    imitation_*.npz             vectors from the warm-start phase
    batch_*.npz                 vectors from each self-play batch
    log.csv                     one line of statistics per batch
```

## 2. Training pipeline

```
          ┌─────────────────────────────────────────────────────────────┐
fresh ──► │ 1. Imitation warm start (only for a new / reset model)       │
model     │    rule-based bots play `imitation_episodes` rounds;        │
          │    `imitation_bid_explore` (30 %) of their announcements    │
          │    are random, so every announcement gets tried.            │
          │    play net: copy the bot's cards (cross-entropy)           │
          │    bid net:  learn the reward of every announcement tried   │
          └─────────────────────────────┬───────────────────────────────┘
                                        ▼
          ┌─────────────────────────────────────────────────────────────┐
          │ 2. Self-play loop, repeated until `episodes` rounds played   │
          │  a) collect `batch_episodes` rounds: the model sits in       │
          │     `learner_seats` random seats, `opponents` fill the rest  │
          │  b) label every decision with the reward of its round        │
          │  c) save the vectors → training_data/<tier>/batch_*.npz      │
          │  d) `epochs` passes of minibatch Adam updates                │
          │  e) save models/<tier>.npz, append to log.csv                │
          └─────────────────────────────────────────────────────────────┘
```

### Why the warm start?

Schafkopf rewards are sparse and noisy: a single round outcome depends on all
32 cards, so a randomly initialised network learns very slowly, and its
random announcements lose so much money that it learns to never announce at
all. Imitating the rule-based bot first gives a sensible starting policy; the
reinforcement learning then improves on it from its own experience.

### Reward

For the seat that made the decisions (all decisions of a round share it):

```
reward = payout_in_cents / reward_scale
       + shaping × (points of own side − 60) / 60          (Sauspiel, Wenz, Solo)
       + shaping × (30 − own points) / 60                  (Ramsch)
```

The payout is the real money result (see [winning-and-scoring.md](winning-and-scoring.md)).
The small shaping term rewards collecting card points even in a lost game and
reduces variance. A thrown-in round gives reward 0.

### Card play: actor-critic policy gradient

For each recorded card decision with state `s`, chosen card `a` and round
reward `R`:

```
A = clip(R − V(s), −3, 3)                                  advantage
L = −A · log π(a|s) + value_coef · (V(s) − R)² − entropy_coef · H(π(·|s))
```

The value head `V(s)` is a learned baseline; moves that turned out better than
expected become more likely, worse ones less likely. Forced moves (only one
legal card) are not recorded. Gradients are clipped to norm 1 and applied with
Adam.

### Bidding: contextual bandit

A bid is a one-shot decision, so the bid net directly predicts the expected
reward `Q(s, a)` of each announcement and is trained by regression on the
outcome of the announcement actually made:

```
L = (Q(s, a) − R)²
```

The AI announces the legal option with the highest Q (pass included). New
networks start pessimistic (`Q(announce) ≈ −1`), and during training
`bid_epsilon` of the bids are random to keep exploring.

### Opponents during training

| `opponents` | Seats not played by the learner                                   |
|-------------|-------------------------------------------------------------------|
| `self`      | all four seats are the learner (pure self-play)                   |
| `heuristic` | rule-based bots with slightly varying bidding courage             |
| `random`    | random bots                                                       |
| `mixed`     | per seat: 50 % rule-based, 35 % current model (not recorded), 15 % random |

## 3. State vectors

All features are relative to the deciding seat: *offset 1* is the next player
clockwise, *offset 2* the one opposite, *offset 3* the previous one. Cards use
the index `suit × 8 + rank` with suits `Eichel, Gras, Herz, Schellen` and ranks
`Sau, Zehn, König, Ober, Unter, Neun, Acht, Sieben` (so `0 = Eichel-Sau`,
`11 = Gras-Ober`, `31 = Schellen-Sieben`).

### Card-play vector (231 floats, `encode_play`)

| Index   | Block                 | Size | Meaning |
|---------|-----------------------|------|---------|
| 0–31    | hand                  | 32 | 1 for each card in my hand |
| 32–63   | played_before         | 32 | cards of the completed tricks |
| 64–95   | trick_by_offset_1     | 32 | card offset 1 played into the current trick |
| 96–127  | trick_by_offset_2     | 32 | card offset 2 played into the current trick |
| 128–159 | trick_by_offset_3     | 32 | card offset 3 played into the current trick |
| 160–191 | trump_cards           | 32 | which cards are trump in this game |
| 192–195 | game_type             | 4  | one-hot Ramsch, Sauspiel, Wenz, Solo |
| 196–199 | contract_suit         | 4  | called suit (Sauspiel) / trump suit (Solo) |
| 200–203 | declarer_offset       | 4  | one-hot seat offset of the declarer (0 = me) |
| 204–209 | relation_offsets_1_3  | 6  | per other seat: [known teammate, known opponent] |
| 210–212 | rufsau_flags          | 3  | I hold the Rufsau, called suit searched, Rufsau revealed |
| 213–216 | position_in_trick     | 4  | one-hot: how many cards are already on the table |
| 217     | trick_number          | 1  | completed tricks / 8 |
| 218–221 | points_by_offset      | 4  | card points taken by each seat / 120 |
| 222–223 | team_points           | 2  | points of my known side, of the known other side / 120 |
| 224–227 | trick_winner_offset   | 4  | who currently wins the trick |
| 228–229 | trick_state           | 2  | points on the table / 40, current winner is mate (+1) / opponent (−1) |
| 230     | trumps_unseen         | 1  | trumps neither in my hand nor played / 14 |

### Bidding vector (76 floats, `encode_bid`)

| Index | Block                  | Size | Meaning |
|-------|------------------------|------|---------|
| 0–31  | hand                   | 32 | my eight cards |
| 32–35 | position_after_dealer  | 4  | one-hot: 1st … 4th to bid |
| 36–39 | highest_bid            | 4  | one-hot: none, Sauspiel, Wenz, Solo |
| 40–54 | earlier_bids           | 15 | per seat offset 1–3: [not yet, pass, Sauspiel, Wenz, Solo] |
| 55–58 | plain_suit_lengths     | 4  | non-trump cards per suit / 6 (Sauspiel trumps) |
| 59–61 | ober_unter_ass         | 3  | number of Ober, Unter, Sau / 4 |
| 62–65 | solo_trump_counts      | 4  | trumps I would have in each Farbsolo / 8 |
| 66    | sauspiel_trumps        | 1  | trumps in a Sauspiel / 8 |
| 67–75 | bid_strengths          | 9  | rule-based hand-strength score of each announcement |

The bid action space (`rules.BID_ACTIONS`): `0 pass, 1–3 Sauspiel Eichel/Gras/Schellen,
4 Wenz, 5–8 Solo Eichel/Gras/Herz/Schellen`.

## 4. Training data files

Every `*.npz` in `training_data/<tier>/` is a numpy archive with:

| Array          | Shape        | dtype   | Content |
|----------------|--------------|---------|---------|
| `bid_states`   | (N_b, 76)    | float32 | bidding vectors |
| `bid_masks`    | (N_b, 9)     | bool    | legal announcements |
| `bid_actions`  | (N_b,)       | int64   | announcement chosen |
| `bid_returns`  | (N_b,)       | float32 | reward of that round |
| `play_states`  | (N_p, 231)   | float32 | card-play vectors |
| `play_masks`   | (N_p, 32)    | bool    | legal cards |
| `play_actions` | (N_p,)       | int64   | card index played |
| `play_returns` | (N_p,)       | float32 | reward of that round |

Load and inspect one yourself:

```python
import numpy as np
d = np.load("training_data/hard/batch_20261101-120000_000001.npz")
print(d["play_states"].shape, d["play_returns"].mean())
```

Only the newest `keep_batches` `batch_*.npz` files are kept; older ones are
deleted automatically. `log.csv` columns: batch, rounds trained so far, mean
reward, mean payout (ct), how often the learner announced, how often it won its
own games, sample counts, losses, entropy, seconds.

## 5. Updating the training by hand

All training is started manually – nothing trains in the background.

### Continue training (keeps what the model already knows)

```bash
uv run schafskopf train --tier medium                  # rounds from config
uv run schafskopf train --tier hard --episodes 100000  # any amount
```

Each call loads `models/<tier>.npz`, plays, learns and saves after every batch.
`Ctrl+C` stops safely; the last saved batch is kept.

### Change hyper-parameters

Edit `config/tiers.toml`, e.g. make medium train against the rule-based bot with a
smaller learning rate:

```toml
[medium]
opponents = "heuristic"
learning_rate = 0.0001
```

and run `train` again. Most keys apply immediately. Changing `hidden`
requires `--reset` because the stored weights no longer fit. One-off overrides
without editing the file: `--opponents`, `--lr`, `--imitation`.

### Start over

```bash
uv run schafskopf train --tier easy --reset            # new network, imitation warm start, RL
uv run schafskopf train --tier easy --reset --imitation 0   # pure RL from scratch
```

### Learn again from the stored vectors

```bash
uv run schafskopf replay --tier hard --epochs 2         # all stored batches
uv run schafskopf replay --tier hard --last 10          # only the newest 10
```

Replay re-uses old experience without playing. Note that the vectors were
produced by older versions of the policy, so replaying many epochs of old
data can make the policy drift; treat it as a fine-tuning tool.

### Check the result

```bash
uv run schafskopf evaluate --tier hard --rounds 5000             # vs rule-based bots
uv run schafskopf evaluate --tier hard --against medium          # tier vs tier
uv run schafskopf evaluate --tier easy --against random
uv run schafskopf data                                           # model + data status
```

`evaluate` seats the tier in seats 1 and 3 and the opponent in seats 2 and 4
and reports the average payout per round and seat (positive = winning money).
Results fluctuate by roughly ±1 ct per round with 4000 rounds.

### Typical workflow

1. `uv run schafskopf train-all` once (a few minutes on a laptop).
2. Play. If an opponent is too strong or weak, change its `temperature` /
   `epsilon` in `config/tiers.toml` (takes effect immediately, no retraining).
3. Want stronger AI? `train --tier hard --episodes 200000`, then `evaluate`.
4. Experimenting? Copy `models/hard.npz` aside first; it is a single file.

## 6. Reference results

Shipped models (trained with `uv run schafskopf train-all --reset --seed 1`,
about 5 minutes on an Apple-silicon laptop), measured with
`uv run schafskopf evaluate … --rounds 4000 --seed 5`. Numbers are cents per
round and seat for the first-named player; each tier uses its table settings
(temperature / epsilon).

| Match-up                    | Result       | Announces | Wins own games |
|-----------------------------|--------------|-----------|----------------|
| hard   vs rule-based bot    | **+0.0 ct**  | 12 %      | 62 %           |
| medium vs rule-based bot    | −6.9 ct      | 18 %      | 40 %           |
| easy   vs rule-based bot    | −22.6 ct     | 25 %      | 27 %           |
| hard   vs medium            | **+5.4 ct**  | 14 %      | 65 %           |
| medium vs easy              | **+15.0 ct** | 18 %      | 53 %           |
| hard   vs easy              | **+21.2 ct** | 12 %      | 74 %           |
| easy   vs random bot        | +2.2 ct      | 31 %      | 40 %           |

During development, pure reinforcement learning from scratch against random
bots improved from +13.8 to +18.5 ct per round within 40 000 rounds, and the
RL phase of the hard tier lifted it from −1.5 ct (after imitation) to parity
with the rule-based bot.

## 7. Limitations

* The networks only see their own hand and public information; they do not
  model card-counting probabilities explicitly – the vectors contain all
  played cards, so the network can learn it.
* Policy-gradient learning on a noisy card game is slow. The trained tiers end
  up around the level of the rule-based bot they started from; more rounds
  (hundreds of thousands) keep improving them only gradually.
* The bid net's Q values are estimates from limited experience – rarely tried
  announcements (e.g. Wenz) are judged less reliably.
