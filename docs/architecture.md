# Architecture

```
schafskopf/
├── pyproject.toml           uv project (Python ≥ 3.14, numpy)
├── config/tiers.toml        hand-editable AI tier settings
├── models/                  trained networks: easy.npz, medium.npz, hard.npz
├── training_data/           training vectors, one folder per tier
│   └── <tier>/batch_*.npz, imitation_*.npz, log.csv
├── docs/                    this documentation
├── tests/                   pytest suite
└── src/schafskopf/
    ├── cards.py             Suit, Rank, Card, deck, dealing
    ├── rules.py             GameType, Bid, Contract, trumps, legal moves, trick winner
    ├── scoring.py           Tariff, winning conditions, payouts
    ├── game.py              Observation, Round (one deal), Match (series), GameSettings
    ├── ui.py                console rendering, input helpers, ConsoleObserver
    ├── cli.py               menus and the `schafskopf` command
    ├── players/
    │   ├── base.py          Player interface
    │   ├── human.py         terminal player (hot-seat aware)
    │   ├── heuristic.py     rule-based bot (+ hand_strength)
    │   └── random_player.py random legal moves
    └── ai/
        ├── config.py        paths and TierConfig loader
        ├── encoding.py      Observation → state vectors (bid / play)
        ├── network.py       numpy MLP with policy and value heads, Adam
        ├── agent.py         ModelBundle (model file) and RLPlayer
        └── training.py      Trainer: imitation, self-play, replay, evaluate
```

## Flow of one round

```
Match.play_round()
 └─ Round.run()
     ├─ run_bidding()   for each seat from dealer+1:  player.bid(observation, legal_bids)
     ├─ run_play()      8 × 4 times:                  player.play(observation, legal_cards)
     ├─ score_round()   → RoundResult (payouts per seat)
     └─ player.round_finished(observation, result)    (learning agents label their moves)
```

* The engine never trusts players: every returned bid/card is checked against
  the legal list.
* Players only ever see an `Observation` – their own hand plus public
  information (bids, tricks, points) and private knowledge (e.g. the Rufsau
  holder knows the partnership). Hidden hands are never passed to a player.
* Observers (`ConsoleObserver`) receive public events for display; the engine
  itself prints nothing, which keeps training fast (~2 ms per round).

## Adding a player type

Subclass `players.base.Player` and implement `bid(obs, legal) -> int` and
`play(obs, legal) -> Card`. Optionally implement `round_finished`. Register it
in `cli.PLAYER_TYPES` / `cli.make_player`.

## Adding a game type

1. Add the type to `rules.GameType` (value = bidding priority) and its
   announcements to `BID_ACTIONS`.
2. Teach `is_trump` / `_trump_ranks` / `trump_order` the trump set.
3. Add scoring in `scoring.score_round`.
4. Update `ai.encoding.GAME_TYPES`; the networks' input/output sizes change,
   so models must be retrained with `--reset`.

## Tests

```bash
uv run pytest
```

covers card points, trump orders, following suit, all Rufsau rules, bidding
legality, Laufende, payouts, full random rounds (zero-sum, 120 points),
encodings, a numeric gradient check of the network and a mini training run.
