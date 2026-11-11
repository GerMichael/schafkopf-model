# How to extend

Recipes for the most common changes. Read [architecture.md](architecture.md)
first for the overall structure. After any change run:

```bash
uv run pytest
```

## 1. Add a player type

1. Create `src/schafskopf/players/<name>.py` with a subclass of
   `players.base.Player`:

   ```python
   class MyPlayer(Player):
       def bid(self, obs: Observation, legal: list[int]) -> int:
           return legal[0]                 # an index from `legal`, into rules.BID_ACTIONS

       def play(self, obs: Observation, legal: list[Card]) -> Card:
           return legal[0]                 # a card from `legal`

       def round_finished(self, obs, result):   # optional
           ...
   ```

2. Export it in `players/__init__.py`.
3. Register it in `cli.py`: append a label to `PLAYER_TYPES` and add a case to
   `make_player` (the `match kind:` index follows the order of `PLAYER_TYPES`;
   keep the existing indices stable).

The engine rejects illegal returns, so a bug in a player cannot corrupt a game.
Use `obs` only: it contains everything the seat may legally know.

## 2. Add a game type

1. `rules.GameType`: add the member (value = bidding priority) and its
   announcements in `BID_ACTIONS` (this fixes `NUM_BIDS`).
2. Teach `rules._trump_ranks`, `is_trump`, `trump_order` (and `effective_suit`
   if needed) the trump set.
3. Add the payout logic in `scoring.score_round` and any new `Tariff` fields.
4. Handle it in `rules.legal_bids` and, if it changes follow-suit rules,
   `rules.legal_cards`.
5. Add the type to `ai.encoding.GAME_TYPES` and update `PLAY_LAYOUT` /
   `BID_LAYOUT` if feature sizes change.
6. Update `docs/game-modes.md` and `docs/rules.md`.
7. Add tests in `tests/test_rules.py` (trump order, legal cards, scoring).
8. **Retrain**: network input/output sizes change, so existing models fail to
   load with a clear error. Run `uv run schafskopf train --tier <tier> --reset`
   for each tier. See §5.

## 3. Change the AI input features

Features live in `ai/encoding.py`. The layout is data: `PLAY_LAYOUT` and
`BID_LAYOUT` are `(name, size)` blocks, and `PLAY_DIM`/`BID_DIM` are derived.

1. Add or change a block in the layout.
2. Fill it in `encode_play` / `encode_bid` **at the position matching the
   layout order** (the code writes sequentially with an index `i`).
3. `tests/test_ai.py::test_layout_sizes` checks that the layout and dimensions
   agree; `test_encodings_in_real_round` checks a real round end to end.
4. Update the layout table in `docs/ai-and-training.md`.
5. Models trained with the old dimension are rejected on load
   ("trained with a different state encoding"); retrain with `--reset`.

## 4. Change the network or the learning rule

* **Size / depth:** edit `hidden` for the tier in `config/tiers.toml` and
  retrain with `--reset` (weights no longer fit).
* **Architecture:** `ai/network.py::PolicyValueNet`. Add layers in `__init__`
  and `forward`. Keep the public numpy-in/numpy-out methods (`predict`,
  `logits`, `train_step`, `q_step`) so agents and trainer keep working.
  If you add parameters, extend `_named_arrays` so they are saved, and bump
  `agent.MODEL_VERSION` when old files become incompatible.
* **Losses:** `train_step` (card play) and `q_step` (bidding). Coefficients
  come from `TierConfig` (`value_coef`, `entropy_coef`, `learning_rate`).
* **Optimizer:** `PolicyValueNet.optimizer` (torch Adam). Another optimizer
  must also be handled in `state_dict_arrays` / `from_state_dict`, otherwise
  its state is not persisted.
* **Rewards:** `agent.round_reward` (payout scaled by `reward_scale` plus a
  card-point shaping term weighted by `shaping`).
* **Device:** the code runs on CPU. For GPU, move the module and tensors in
  `_infer`/`train_step` and convert back with `.cpu().numpy()`; for these small
  networks single-state inference is usually faster on CPU.

## 5. Add or tune a difficulty tier

* **Tune** an existing tier by editing `config/tiers.toml`; every key is
  documented at the top of the file. `temperature` and `epsilon` (playing
  strength) take effect immediately; training keys apply to the next
  `train` call.
* **Add** a tier: add a `[name]` table to the TOML, add the name to
  `ai.config.TIERS`, and add the entry to `cli.PLAYER_TYPES` / `make_player`
  (`case 1 | 2 | 3` maps to `TIERS[kind - 1]`; update the mapping so indices
  stay consistent).
* **Train:**

  ```bash
  uv run schafskopf train --tier medium            # continue from the saved model
  uv run schafskopf train --tier medium --reset    # start over (re-runs imitation warm start)
  ```

  Copy `models/<tier>.npz` aside before experiments; it is a single file.
* **Check strength:** `uv run schafskopf evaluate --tier medium --against heuristic`
  (`--against` also accepts `random` or another tier). `train-all` trains all
  three tiers in one go.

## 6. Add a CLI command

`cli.build_parser` defines the sub-commands (argparse); `cli.main` dispatches.
Add the parser entry and a function, mirroring `evaluate_tier` or
`replay_tier`. Interactive menu entries are in `main_menu` / `training_menu`.

## 7. Change the scoring tariff

`scoring.Tariff` holds every amount in cents. Change the defaults, or pass a
custom `Tariff` through `GameSettings(tariff=...)`. Tests in
`tests/test_rules.py` that assert concrete payouts assume the default tariff.

## 8. Checklist before opening a PR

* `uv run pytest` passes (CI runs the same with `uv sync --locked`).
* If you changed dependencies: `uv lock` and commit `uv.lock`.
* If features, game types or architecture changed: models retrained or the
  incompatibility noted in [known-issues.md](known-issues.md).
* Docs updated (`rules.md`, `game-modes.md`, `ai-and-training.md`,
  `architecture.md` as applicable).
