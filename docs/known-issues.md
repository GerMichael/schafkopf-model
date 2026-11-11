# Known issues and limitations

Status of the project as of the PyTorch migration of the AI. Items marked
**(unverified)** were identified by reading the code and have not been
reproduced.

## AI strength and learning

* **Strength plateau.** Policy-gradient learning on a noisy card game is slow.
  The trained tiers end up around the level of the rule-based bot they were
  warm-started from; further rounds improve them only gradually
  (see [ai-and-training.md §7](ai-and-training.md)).
* **No explicit hidden-information model.** The networks see their own hand
  and all played cards, but there is no belief tracking or search over the
  opponents' possible hands. Card counting must be learned implicitly.
* **Bid network is poorly calibrated for rare games.** Q values come from
  limited experience, so rarely tried announcements (e.g. Wenz, Solo) are
  judged less reliably. The bid head is initialised pessimistically
  (announcing assumed to cost money).
* **Bid decision is a one-step bandit.** Bidding is trained with the final
  round reward only, ignoring that later bidders react to earlier ones beyond
  what is encoded in the state.
* **Value head unused for bidding.** The bid network trains only the action
  head; its value head is never trained.
* **Training signal depends on opponents.** Tiers trained against the rule-based
  bot may overfit its style and generalise worse to humans. `mixed` mitigates
  this but does not remove it.

## PyTorch migration

* **Old models are not bit-identical.** Models trained with the earlier numpy
  implementation load correctly (same `.npz` layout, Adam state included), but
  PyTorch computes in float32 with different operation order. Predictions can
  differ in the last decimals, and resumed training will not reproduce the
  exact numpy trajectory.
* **Model files keep the `.npz` extension.** They are numpy archives, not
  `torch.save` checkpoints, so `torch.load` cannot read them. This was done for
  backward compatibility.
* **Seeding is global.** `PolicyValueNet(seed=...)` calls `torch.manual_seed`,
  which affects the global PyTorch RNG. Creating a net mid-run therefore
  reseeds other torch consumers. Two nets created with the same seed get
  identical initial weights. (`ModelBundle.fresh` offsets the play-net seed by
  one for this reason.)
* **Reproducibility is CPU-only and single-threaded.** `torch.set_num_threads(1)`
  is set at import time of `ai/network.py` for the whole process. This is
  deliberate (tiny networks, faster single-state inference) but affects any other
  torch code in the same process.
* **CPU only.** There is no GPU support; moving to a device is a manual change
  (see [how-to-extend.md §4](how-to-extend.md)).
* **Heavy dependency.** `torch` adds a large install (hundreds of MB). On Linux
  CI runners the default wheel may pull CUDA libraries, which makes
  `uv sync` slow. **(unverified)** A CPU-only index in `pyproject.toml`
  (`[tool.uv.sources]`) would fix this.
* **Python ≥ 3.14 requirement.** `requires-python = ">=3.14"` is strict; the
  project cannot be installed on older interpreters, and it depends on PyTorch
  publishing wheels for the interpreter version in use.

## Persistence

* **Resume after interruption loses the current batch.** The model is
  checkpointed per batch; the batch in progress is lost on Ctrl-C.
* **No cross-version migration.** Changing the encoding (`PLAY_DIM`/`BID_DIM`)
  or the number of announcements makes old models unloadable; retraining with
  `--reset` is the only path.
* **Optimizer-state fidelity.** If an `.npz` has `steps == 0` the optimizer
  starts fresh. Adam state is restored per parameter, but a non-Adam optimizer
  would silently not be persisted.
* **Models and vectors are not in git.** `.gitignore` excludes `*.npz`, so a
  fresh clone (and CI) has no trained models; `make_ai` then falls back to the
  rule-based bot with a warning. Only `training_data/<tier>/log.csv` is tracked.
  Back up `models/` separately.
* **Training data grows.** `keep_batches` prunes old `batch_*.npz`, but
  `imitation_*.npz` files and `log.csv` are never removed.
* **Stray `.DS_Store` files** (including `training_data/.DS_Store`) are tracked
  in git and should be removed and ignored.

## Game rules

Not implemented (see [rules.md](rules.md)): Kontra/Re, Klopfen/Stoß, Tout, Sie,
Geier, Bettel, Farbwenz and the short 24-card deck. Local variants differ; the
tariff is configurable (`Tariff`) but other house rules are not.

## Interface and engine

* **Console only.** There is no GUI or network play. Hot-seat play with several
  humans relies on terminal screen clearing.
* **No game save/resume.** A `Match` exists only in memory.
* **Engine speed.** Pure Python with Python-level rule checks; fine for
  training in the range of tens of thousands of rounds, but not optimised for
  millions.

## Tests

* **No statistical strength test.** Tests check shapes, masks, gradients'
  effects and that a mini training run completes, not that a model plays well.
  Strength is checked manually with `evaluate`.
* **The numeric gradient check was removed** with the move to autograd, since
  PyTorch computes the gradients. The remaining tests verify that
  `train_step`/`q_step` reduce their losses and respect the legal-move mask.
* **Test startup is slower** than before because the AI tests import PyTorch.

## Documentation

* Some wording in older docs may still refer to "numpy networks"; the AI
  implementation is PyTorch. Report or fix instances you find.
