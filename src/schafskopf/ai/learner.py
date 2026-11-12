"""Losses and optimisation for ``PolicyValueNet``.

Card play (actor-critic) – loss for one sample with return R, chosen action a
and advantage A = clip(R - V(s))::

    L = -A · log π(a|s)  +  c_v · (V(s) - R)²  -  c_e · H(π(·|s))

Bidding (contextual bandit) – the action head predicts Q(s, a)::

    L = (Q(s, a) - R)²
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .network import PolicyValueNet, masked_logits


@dataclass(frozen=True)
class LossConfig:
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    adv_clip: float = 3.0


def actor_critic_loss(logits: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor,
                      returns: torch.Tensor, cfg: LossConfig, supervised: bool = False
                      ) -> tuple[torch.Tensor, dict[str, float]]:
    """With ``supervised=True`` the advantage is fixed to 1, i.e. plain cross-entropy (imitation)."""
    dist = Categorical(logits=masked_logits(logits, mask))
    if supervised:
        adv = torch.ones_like(value)
    else:
        adv = (returns - value).detach().clamp(-cfg.adv_clip, cfg.adv_clip)
    policy_loss = -(adv * dist.log_prob(actions)).mean()
    value_loss = ((value - returns) ** 2).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
    return loss, {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item()}


def q_loss(q: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Squared error on the output of the action that was actually taken."""
    taken = q.gather(1, actions[:, None]).squeeze(1)
    loss = ((taken - returns) ** 2).mean()
    return loss, {"policy_loss": 0.0, "value_loss": loss.item(), "entropy": 0.0}


class Learner:
    """Owns the optimizer for one net: Adam, global gradient-norm clipping, loss configuration."""

    def __init__(self, net: PolicyValueNet, lr: float = 1e-3, loss: LossConfig = LossConfig()) -> None:
        self.net = net
        self.loss = loss
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _tensors(self, x, actions, returns):
        dtype = self.net.policy_head.weight.dtype
        return (torch.as_tensor(x, dtype=dtype), torch.as_tensor(actions, dtype=torch.long),
                torch.as_tensor(returns, dtype=dtype))

    def _step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.loss.max_grad_norm)
        self.optimizer.step()

    def train_step(self, x: np.ndarray, mask: np.ndarray, actions: np.ndarray, returns: np.ndarray,
                   supervised: bool = False) -> dict[str, float]:
        """One actor-critic step on a minibatch; returns loss statistics."""
        xt, at, rt = self._tensors(x, actions, returns)
        logits, value = self.net(xt)
        loss, stats = actor_critic_loss(logits, value, torch.as_tensor(mask, dtype=torch.bool), at, rt,
                                        self.loss, supervised)
        self._step(loss)
        return stats

    def q_step(self, x: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> dict[str, float]:
        """One Q-regression step (bidding); the value head is not trained."""
        xt, at, rt = self._tensors(x, actions, returns)
        loss, stats = q_loss(self.net(xt)[0], at, rt)
        self._step(loss)
        return stats
