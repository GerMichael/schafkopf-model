"""Self-play reinforcement learning (policy gradient with a learned baseline).

One training call repeats this loop until the requested number of rounds is
played:

1. **Collect** ``batch_episodes`` rounds. The learning model sits in
   ``learner_seats`` random seats; the other seats are filled according to
   ``opponents``. Every decision of a learner seat is stored as a vector.
2. **Label** each decision with the Monte-Carlo return of its round (the
   reward of the seat that made it – all moves of a round share it).
3. **Save** the batch as ``training_data/<tier>/batch_*.npz``.
4. **Update** both networks for ``epochs`` passes over the batch.
5. **Checkpoint** the model to ``models/<tier>.npz`` and append a line to
   ``training_data/<tier>/log.csv``.

``replay`` re-trains on the vector files already on disk without playing.
"""

from __future__ import annotations

import csv
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..game import GameSettings, Round
from ..players.base import Player
from ..players.heuristic import HeuristicPlayer
from ..players.random_player import RandomPlayer
from .agent import ModelBundle, RLPlayer, Trajectory, round_reward
from .config import TierConfig, data_dir, model_path
from .encoding import BID_DIM, PLAY_DIM, bid_mask, card_mask, encode_bid, encode_play
from .network import PolicyValueNet

BATCH_KEYS = ("bid_states", "bid_masks", "bid_actions", "bid_returns",
              "play_states", "play_masks", "play_actions", "play_returns")


@dataclass
class BatchStats:
    batch: int
    episodes: int
    mean_reward: float
    mean_payout: float
    declare_rate: float
    declarer_win_rate: float
    bid_samples: int
    play_samples: int
    policy_loss: float
    value_loss: float
    entropy: float
    bid_loss: float
    seconds: float


def trajectories_to_arrays(records: list[tuple[Trajectory, float, int, bool]]) -> dict[str, np.ndarray]:
    out: dict[str, list] = {k: [] for k in BATCH_KEYS}
    for traj, reward, _, _ in records:
        out["bid_states"] += traj.bid_states
        out["bid_masks"] += traj.bid_masks
        out["bid_actions"] += traj.bid_actions
        out["bid_returns"] += [reward] * len(traj.bid_actions)
        out["play_states"] += traj.play_states
        out["play_masks"] += traj.play_masks
        out["play_actions"] += traj.play_actions
        out["play_returns"] += [reward] * len(traj.play_actions)
    dims = {"bid": (BID_DIM, 9), "play": (PLAY_DIM, 32)}
    arrays: dict[str, np.ndarray] = {}
    for head, (dim, n_actions) in dims.items():
        arrays[f"{head}_states"] = np.array(out[f"{head}_states"], dtype=np.float32).reshape(-1, dim)
        arrays[f"{head}_masks"] = np.array(out[f"{head}_masks"], dtype=bool).reshape(-1, n_actions)
        arrays[f"{head}_actions"] = np.array(out[f"{head}_actions"], dtype=np.int64)
        arrays[f"{head}_returns"] = np.array(out[f"{head}_returns"], dtype=np.float32)
    return arrays


def update_net(net: PolicyValueNet, states: np.ndarray, masks: np.ndarray, actions: np.ndarray,
               returns: np.ndarray, cfg: TierConfig, rng: np.random.Generator,
               lr: float | None = None, epochs: int | None = None,
               supervised: bool = False, q_learning: bool = False) -> dict[str, float]:
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n, steps = len(states), 0
    if n == 0:
        return stats
    for _ in range(epochs or cfg.epochs):
        order = rng.permutation(n)
        for start in range(0, n, cfg.minibatch):
            idx = order[start:start + cfg.minibatch]
            if q_learning:
                s = net.q_step(states[idx], actions[idx], returns[idx], lr or cfg.learning_rate)
            else:
                s = net.train_step(states[idx], masks[idx], actions[idx], returns[idx],
                                   lr or cfg.learning_rate, cfg.value_coef, cfg.entropy_coef,
                                   supervised=supervised)
            for k in stats:
                stats[k] += s[k]
            steps += 1
    return {k: v / steps for k, v in stats.items()}


class ImitationRecorder(Player):
    """Wraps a teacher player and records its decisions as training vectors."""

    def __init__(self, teacher: Player, recorder: list, reward_scale: float, shaping: float,
                 bid_explore: float = 0.0, rng: random.Random | None = None) -> None:
        super().__init__(teacher.name)
        self.teacher = teacher
        self.bid_explore = bid_explore
        self.rng = rng or random.Random()
        self.recorder = recorder
        self.reward_scale = reward_scale
        self.shaping = shaping
        self._traj = Trajectory()

    def bid(self, obs, legal):
        # Sometimes try a random announcement so the bid network sees what
        # every option is worth, not only the ones the teacher likes.
        if self.rng.random() < self.bid_explore:
            action = self.rng.choice(legal)
        else:
            action = self.teacher.bid(obs, legal)
        self._traj.bid_states.append(encode_bid(obs))
        self._traj.bid_masks.append(bid_mask(legal))
        self._traj.bid_actions.append(action)
        return action

    def play(self, obs, legal):
        card = self.teacher.play(obs, legal)
        if len(legal) > 1:
            self._traj.play_states.append(encode_play(obs))
            self._traj.play_masks.append(card_mask(legal))
            self._traj.play_actions.append(card.index)
        return card

    def round_finished(self, obs, result):
        reward = round_reward(obs, result, self.reward_scale, self.shaping)
        payout = result.payouts[obs.seat] if result else 0
        declarer = bool(result and result.contract.declarer == obs.seat)
        self.recorder.append((self._traj, reward, payout, declarer))
        self._traj = Trajectory()


class Trainer:
    def __init__(self, cfg: TierConfig, reset: bool = False, seed: int | None = None,
                 opponents: str | None = None, learning_rate: float | None = None,
                 save_data: bool = True, settings: GameSettings = GameSettings()) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.opponents = opponents or cfg.opponents
        self.learning_rate = learning_rate or cfg.learning_rate
        self.save_data = save_data
        self.settings = settings
        self.model_file = model_path(cfg.name)
        self.data_dir = data_dir(cfg.name)
        if self.model_file.exists() and not reset:
            self.model = ModelBundle.load(self.model_file)
            if tuple(self.model.meta.get("hidden", cfg.hidden)) != tuple(cfg.hidden):
                raise ValueError(f"{self.model_file} has hidden layers {self.model.meta['hidden']} but "
                                 f"config says {list(cfg.hidden)}; use --reset to start over")
        else:
            self.model = ModelBundle.fresh(cfg, seed)

    # -- collection -------------------------------------------------------
    def _opponent(self, seat: int) -> Player:
        kind = self.opponents
        if kind == "mixed":
            kind = self.rng.choices(["heuristic", "self", "random"], weights=[0.5, 0.35, 0.15])[0]
        if kind == "heuristic":
            return HeuristicPlayer(f"H{seat}", self.rng, threshold=self.rng.uniform(0.9, 1.15))
        if kind == "random":
            return RandomPlayer(f"R{seat}", self.rng)
        # "self": the current model, not recorded, playing a bit stochastically
        return RLPlayer(f"S{seat}", self.model, temperature=0.5, rng=self.rng)

    def collect(self, episodes: int) -> list[tuple[Trajectory, float, int, bool]]:
        records: list = []
        learners = max(1, min(4, self.cfg.learner_seats))
        if self.opponents == "self":
            learners = 4
        for i in range(episodes):
            learner_seats = set(self.rng.sample(range(4), learners))
            players = [
                RLPlayer(f"L{s}", self.model, temperature=1.0, rng=self.rng, recorder=records,
                         reward_scale=self.cfg.reward_scale, shaping=self.cfg.shaping,
                         bid_epsilon=self.cfg.bid_epsilon, bid_temperature=0.0)
                if s in learner_seats else self._opponent(s)
                for s in range(4)
            ]
            Round(players, self.rng.randrange(4), self.settings, self.rng).run()
        return records

    # -- persistence of vectors ------------------------------------------
    def save_batch(self, arrays: dict[str, np.ndarray], prefix: str = "batch") -> Path | None:
        if not self.save_data:
            return None
        self.data_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        path = self.data_dir / f"{prefix}_{stamp}_{self.model.meta['batches']:06d}.npz"
        np.savez_compressed(path, **arrays)
        batches = sorted(self.data_dir.glob("batch_*.npz"))
        for old in batches[: max(0, len(batches) - self.cfg.keep_batches)]:
            old.unlink()
        return path

    def _log(self, stats: BatchStats) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log = self.data_dir / "log.csv"
        new = not log.exists()
        with log.open("a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(list(stats.__dataclass_fields__))
            writer.writerow([round(v, 5) if isinstance(v, float) else v for v in stats.__dict__.values()])

    # -- main loop --------------------------------------------------------
    def learn(self, arrays: dict[str, np.ndarray], epochs: int | None = None,
              supervised: bool = False) -> dict[str, float]:
        bid = update_net(self.model.bid_net, arrays["bid_states"], arrays["bid_masks"],
                         arrays["bid_actions"], arrays["bid_returns"], self.cfg, self.np_rng,
                         self.learning_rate, epochs, q_learning=True)
        play = update_net(self.model.play_net, arrays["play_states"], arrays["play_masks"],
                          arrays["play_actions"], arrays["play_returns"], self.cfg, self.np_rng,
                          self.learning_rate, epochs, supervised)
        return {**play, "bid_loss": bid["value_loss"]}

    def imitate(self, episodes: int, epochs: int = 5,
                progress: Callable[[int, dict[str, float]], None] | None = None) -> ModelBundle:
        """Warm start: learn to copy the heuristic player (behaviour cloning).

        The value heads are trained on the observed returns at the same time,
        so the RL phase starts with a sensible baseline.
        """
        records: list = []
        for _ in range(episodes):
            players = [ImitationRecorder(HeuristicPlayer(f"T{s}", self.rng), records,
                                         self.cfg.reward_scale, self.cfg.shaping,
                                         self.cfg.imitation_bid_explore, self.rng) for s in range(4)]
            Round(players, self.rng.randrange(4), self.settings, self.rng).run()
        arrays = trajectories_to_arrays(records)
        self.save_batch(arrays, prefix="imitation")
        losses = self.learn(arrays, epochs, supervised=True)
        self.model.meta["imitation_episodes"] = self.model.meta.get("imitation_episodes", 0) + episodes
        self.model.save(self.model_file)
        if progress:
            progress(len(records), losses)
        return self.model

    def train(self, episodes: int | None = None,
              progress: Callable[[BatchStats], None] | None = None,
              on_imitation: Callable[[int, dict[str, float]], None] | None = None) -> ModelBundle:
        remaining = episodes or self.cfg.episodes
        fresh = not self.model.meta.get("episodes") and not self.model.meta.get("imitation_episodes")
        if fresh and self.cfg.imitation_episodes > 0:
            self.imitate(self.cfg.imitation_episodes, self.cfg.imitation_epochs, on_imitation)
        while remaining > 0:
            started = time.time()
            n = min(self.cfg.batch_episodes, remaining)
            records = self.collect(n)
            arrays = trajectories_to_arrays(records)
            self.save_batch(arrays)
            losses = self.learn(arrays)
            remaining -= n
            self.model.meta["episodes"] = self.model.meta.get("episodes", 0) + n
            self.model.meta["batches"] = self.model.meta.get("batches", 0) + 1
            self.model.save(self.model_file)

            decl = [r for r in records if r[3]]
            stats = BatchStats(
                batch=self.model.meta["batches"],
                episodes=self.model.meta["episodes"],
                mean_reward=float(np.mean([r[1] for r in records])),
                mean_payout=float(np.mean([r[2] for r in records])),
                declare_rate=len(decl) / max(1, len(records)),
                declarer_win_rate=sum(r[2] > 0 for r in decl) / max(1, len(decl)),
                bid_samples=len(arrays["bid_actions"]),
                play_samples=len(arrays["play_actions"]),
                seconds=time.time() - started,
                **losses,
            )
            self._log(stats)
            if progress:
                progress(stats)
        return self.model

    def replay(self, epochs: int = 1, limit: int | None = None) -> tuple[int, dict[str, float]]:
        """Train on the vector batches stored in training_data/<tier>/."""
        files = sorted(self.data_dir.glob("batch_*.npz"))
        if limit:
            files = files[-limit:]
        if not files:
            return 0, {}
        parts: dict[str, list[np.ndarray]] = {k: [] for k in BATCH_KEYS}
        for path in files:
            with np.load(path) as data:
                for k in BATCH_KEYS:
                    parts[k].append(data[k])
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        losses = self.learn(arrays, epochs)
        self.model.meta["replays"] = self.model.meta.get("replays", 0) + 1
        self.model.save(self.model_file)
        return len(files), losses


def evaluate(model: ModelBundle, rounds: int = 1000, against: str = "heuristic", seed: int | None = None,
             opponent_model: ModelBundle | None = None, temperature: float = 0.0, epsilon: float = 0.0,
             opponent_temperature: float = 0.0, opponent_epsilon: float = 0.0) -> dict[str, float]:
    """Model in seats 0 and 2 against two opponents in seats 1 and 3.

    Returns the average payout per round (cents) of the model seats and
    some auxiliary statistics.
    """
    rng = random.Random(seed)
    totals = [0, 0, 0, 0]
    declared = won = 0

    def opponent(seat: int) -> Player:
        if against == "random":
            return RandomPlayer(f"R{seat}", rng)
        if against == "model" and opponent_model is not None:
            return RLPlayer(f"O{seat}", opponent_model, opponent_temperature, opponent_epsilon, rng=rng)
        return HeuristicPlayer(f"H{seat}", rng)

    players = [RLPlayer("A0", model, temperature, epsilon, rng=rng), opponent(1),
               RLPlayer("A2", model, temperature, epsilon, rng=rng), opponent(3)]
    for i in range(rounds):
        result = Round(players, i % 4, rng=rng).run()
        if result is None:
            continue
        for s in range(4):
            totals[s] += result.payouts[s]
        if result.contract.declarer in (0, 2):
            declared += 1
            won += result.declarers_won
    return {
        "rounds": rounds,
        "model_cents_per_round": (totals[0] + totals[2]) / (2 * rounds),
        "opponent_cents_per_round": (totals[1] + totals[3]) / (2 * rounds),
        "model_declare_rate": declared / (2 * rounds),
        "model_declarer_win_rate": won / max(1, declared),
    }
