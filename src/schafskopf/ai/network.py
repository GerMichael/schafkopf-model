"""The policy/value network (architecture only; training lives in ``learner.py``).

Architecture (shared trunk, two heads)::

    x ─ Linear ─ ReLU ─ Linear ─ ReLU ─┬─ Linear → action logits (masked softmax)
                                       └─ Linear → state value V(s)

The bidding net uses the action head as Q(s, a) and ignores the value head.
The public inference interface (``predict`` / ``logits``) is numpy-in / numpy-out.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

NEG_INF = -1e9


def masked_logits(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Temperature-scaled logits with illegal actions pushed to ``NEG_INF`` (softmax → probability 0).

    Temperatures < 1 sharpen the distribution, > 1 flatten it.
    """
    return (logits / max(temperature, 1e-6)).masked_fill(~mask, NEG_INF)


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden: Sequence[int] = (128, 64),
                 seed: int | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden = tuple(int(h) for h in hidden)

        # fork the RNG so a seed makes the init reproducible without touching global torch state
        with (torch.random.fork_rng() if seed is not None else nullcontext()):
            if seed is not None:
                torch.manual_seed(seed)
            sizes = (input_dim, *self.hidden)
            layers: list[nn.Module] = []
            for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
                layers += [nn.Linear(fan_in, fan_out), nn.ReLU()]
            self.trunk = nn.Sequential(*layers)
            self.policy_head = nn.Linear(sizes[-1], num_actions)
            self.value_head = nn.Linear(sizes[-1], 1)
            self._init_weights()

    def _init_weights(self) -> None:
        """He init for the trunk, small init for the heads, zero biases."""
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        for head in (self.policy_head, self.value_head):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    @property
    def config(self) -> dict:
        """Constructor arguments needed to rebuild this architecture."""
        return {"input_dim": self.input_dim, "num_actions": self.num_actions, "hidden": list(self.hidden)}

    @classmethod
    def from_config(cls, config: dict) -> PolicyValueNet:
        return cls(config["input_dim"], config["num_actions"], config["hidden"])

    def set_action_bias(self, index: slice | int | list[int], value: float) -> None:
        """Set entries of the action-head bias (e.g. to start with pessimistic Q-values)."""
        with torch.no_grad():
            self.policy_head.bias[index] = value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch ``x`` of shape (n, input_dim) → (action logits (n, A), value (n,))."""
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    # -- numpy inference --------------------------------------------------
    def _infer(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self(torch.as_tensor(x, dtype=self.policy_head.weight.dtype)[None, :])

    def predict(self, x: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> tuple[np.ndarray, float]:
        """Probabilities over legal actions and the value for a single state vector."""
        logits, value = self._infer(x)
        probs = torch.softmax(masked_logits(logits, torch.as_tensor(mask)[None, :], temperature), dim=-1)
        return probs[0].numpy(), float(value[0])

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Raw (unmasked) action-head output for a single state vector (e.g. bidding Q-values)."""
        return self._infer(x)[0][0].numpy()
