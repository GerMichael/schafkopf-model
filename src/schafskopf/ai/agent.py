"""The reinforcement-learning player and its model file."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..cards import Card
from ..game import Observation
from ..players.base import Player
from ..rules import NUM_BIDS, GameType
from ..scoring import RoundResult
from .config import TierConfig, model_path
from .encoding import BID_DIM, PLAY_DIM, bid_mask, card_mask, encode_bid, encode_play
from .learner import Learner, LossConfig
from .network import PolicyValueNet

MODEL_VERSION = 1


@dataclass
class ModelBundle:
    """Bidding and card-play networks (with their learners) of one tier, stored in models/<tier>.pt."""

    bid_net: PolicyValueNet
    play_net: PolicyValueNet
    meta: dict = field(default_factory=dict)
    bid_learner: Learner = field(init=False)
    play_learner: Learner = field(init=False)

    def __post_init__(self) -> None:
        self.bid_learner = Learner(self.bid_net)
        self.play_learner = Learner(self.play_net)

    def configure(self, lr: float, loss: LossConfig) -> None:
        """Set learning rate and loss hyperparameters for both learners."""
        for learner in (self.bid_learner, self.play_learner):
            learner.set_lr(lr)
            learner.loss = loss

    @classmethod
    def fresh(cls, tier: TierConfig, seed: int | None = None) -> ModelBundle:
        meta = {"version": MODEL_VERSION, "tier": tier.name, "episodes": 0, "batches": 0,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "hidden": list(tier.hidden)}
        bid_net = PolicyValueNet(BID_DIM, NUM_BIDS, tier.hidden, seed)
        # Pessimistic start: announcing a game is assumed to cost money until
        # experience shows otherwise (the bid head predicts expected rewards).
        bid_net.set_action_bias(slice(1, None), -1.0)
        play_net = PolicyValueNet(PLAY_DIM, 32, tier.hidden, None if seed is None else seed + 1)
        return cls(bid_net, play_net, meta)

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        ckpt = torch.load(path, weights_only=True)
        nets = {k: PolicyValueNet.from_config(ckpt[k]["arch"]) for k in ("bid", "play")}
        for k, net in nets.items():
            net.load_state_dict(ckpt[k]["model"])
        bundle = cls(nets["bid"], nets["play"], ckpt["meta"])
        bundle.bid_learner.optimizer.load_state_dict(ckpt["bid"]["optimizer"])
        bundle.play_learner.optimizer.load_state_dict(ckpt["play"]["optimizer"])
        if bundle.bid_net.input_dim != BID_DIM or bundle.play_net.input_dim != PLAY_DIM:
            raise ValueError(f"{path} was trained with a different state encoding; retrain with --reset")
        return bundle

    @classmethod
    def load_tier(cls, tier: str) -> ModelBundle:
        return cls.load(model_path(tier))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.meta["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        ckpt = {"meta": self.meta}
        for key, net, learner in (("bid", self.bid_net, self.bid_learner), ("play", self.play_net, self.play_learner)):
            ckpt[key] = {"arch": net.config, "model": net.state_dict(), "optimizer": learner.optimizer.state_dict()}
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)


def choose(probs: np.ndarray, mask: np.ndarray, temperature: float, epsilon: float,
           rng: random.Random, logits: np.ndarray | None = None) -> int:
    legal = np.flatnonzero(mask)
    if epsilon and rng.random() < epsilon:
        return int(rng.choice(legal))
    if temperature <= 0:
        return int(legal[np.argmax(logits[legal] if logits is not None else probs[legal])])
    return int(rng.choices(range(len(probs)), weights=probs.tolist())[0])


@dataclass
class Trajectory:
    """Decisions of one seat in one round, waiting for the final reward."""

    bid_states: list[np.ndarray] = field(default_factory=list)
    bid_masks: list[np.ndarray] = field(default_factory=list)
    bid_actions: list[int] = field(default_factory=list)
    play_states: list[np.ndarray] = field(default_factory=list)
    play_masks: list[np.ndarray] = field(default_factory=list)
    play_actions: list[int] = field(default_factory=list)


def round_reward(obs: Observation, result: RoundResult | None, reward_scale: float, shaping: float) -> float:
    """Reward of one seat: payout plus a small card-point shaping term."""
    if result is None:
        return 0.0
    reward = result.payouts[obs.seat] / reward_scale
    if result.contract.game_type is GameType.RAMSCH:
        reward += shaping * (30 - result.points[obs.seat]) / 60
    else:
        team = result.declarer_team if obs.seat in result.declarer_team else \
            tuple(s for s in range(4) if s not in result.declarer_team)
        reward += shaping * (sum(result.points[s] for s in team) - 60) / 60
    return reward


class RLPlayer(Player):
    """Plays with a trained :class:`ModelBundle`.

    Bidding: the bid network estimates the expected reward Q(s, a) of every
    legal announcement; the best one is chosen (or sampled from
    ``softmax(Q / temperature)``; during training ``bid_epsilon`` random bids
    are mixed in for exploration). Card play: the play network outputs a
    policy π(card | s), played greedily or sampled with ``temperature``.

    With a ``recorder`` attached (during training) every decision is stored as
    a (state vector, legal mask, action) triple and labelled with the round's
    reward when the round is over.
    """

    def __init__(self, name: str, model: ModelBundle, temperature: float = 0.0, epsilon: float = 0.0,
                 rng: random.Random | None = None, recorder: list | None = None,
                 reward_scale: float = 50.0, shaping: float = 0.3, tier: str = "",
                 bid_epsilon: float | None = None, bid_temperature: float | None = None) -> None:
        super().__init__(name)
        self.model = model
        self.temperature = temperature
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.recorder = recorder
        self.reward_scale = reward_scale
        self.shaping = shaping
        self.tier = tier
        self.bid_epsilon = epsilon if bid_epsilon is None else bid_epsilon
        self.bid_temperature = temperature if bid_temperature is None else bid_temperature
        self._traj = Trajectory()

    @property
    def kind(self) -> str:
        return f"AI ({self.tier})" if self.tier else "AI"

    def bid(self, obs: Observation, legal: list[int]) -> int:
        x, mask = encode_bid(obs), bid_mask(legal)
        net = self.model.bid_net
        q = net.logits(x)
        probs, _ = net.predict(x, mask, max(self.bid_temperature, 1e-3))
        action = choose(probs, mask, self.bid_temperature, self.bid_epsilon, self.rng, q)
        if self.recorder is not None:
            self._traj.bid_states.append(x)
            self._traj.bid_masks.append(mask)
            self._traj.bid_actions.append(action)
        return action

    def play(self, obs: Observation, legal: list[Card]) -> Card:
        if len(legal) == 1 and self.recorder is None:
            return legal[0]
        x, mask = encode_play(obs), card_mask(legal)
        net = self.model.play_net
        probs, _ = net.predict(x, mask, max(self.temperature, 1e-3))
        action = choose(probs, mask, self.temperature, self.epsilon, self.rng, net.logits(x))
        if self.recorder is not None and len(legal) > 1:
            self._traj.play_states.append(x)
            self._traj.play_masks.append(mask)
            self._traj.play_actions.append(action)
        return Card.from_index(action)

    def round_finished(self, obs: Observation, result: RoundResult | None) -> None:
        if self.recorder is not None:
            reward = round_reward(obs, result, self.reward_scale, self.shaping)
            payout = result.payouts[obs.seat] if result else 0
            declarer = bool(result and result.contract.declarer == obs.seat)
            self.recorder.append((self._traj, reward, payout, declarer))
        self._traj = Trajectory()
