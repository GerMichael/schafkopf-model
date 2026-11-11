"""A small policy/value neural network built with PyTorch, trained with Adam.

Architecture (shared trunk, two heads)::

    x ─ Linear ─ ReLU ─ Linear ─ ReLU ─┬─ Linear → action logits (masked softmax)
                                       └─ Linear → state value V(s)

Card play (actor-critic) – loss for one sample with return R, chosen action a
and advantage A = clip(R - V(s))::

    L = -A · log π(a|s)  +  c_v · (V(s) - R)²  -  c_e · H(π(·|s))

Bidding (contextual bandit) – the action head predicts Q(s, a)::

    L = (Q(s, a) - R)²

The public interface takes and returns numpy arrays, so agents and the trainer
do not need to know about torch. Weights are persisted in the same ``.npz``
layout as before (weights stored as ``(in, out)`` matrices).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

NEG_INF = -1e9

torch.set_num_threads(1)  # tiny MLPs: threading overhead only slows single-state inference down


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden: Sequence[int] = (128, 64),
                 seed: int | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden = tuple(int(h) for h in hidden)
        if seed is not None:
            torch.manual_seed(seed)
        sizes = (input_dim, *self.hidden)
        self.trunk = nn.ModuleList(nn.Linear(sizes[k], sizes[k + 1]) for k in range(len(self.hidden)))
        self.policy_head = nn.Linear(sizes[-1], num_actions)
        self.value_head = nn.Linear(sizes[-1], 1)
        for layer in self.trunk:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        for head in (self.policy_head, self.value_head):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.steps = 0

    # -- inference --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.trunk:
            h = torch.relu(layer(h))
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @staticmethod
    def masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        z = np.where(mask, logits / max(temperature, 1e-6), NEG_INF)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z) * mask
        return e / e.sum(axis=-1, keepdims=True)

    @torch.no_grad()
    def _infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, value = self(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, :])
        return logits[0].numpy(), value.numpy()

    def predict(self, x: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> tuple[np.ndarray, float]:
        """Action probabilities and value for a single state vector."""
        logits, value = self._infer(x)
        return self.masked_softmax(logits[None, :], mask[None, :], temperature)[0], float(value[0])

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._infer(x)[0]

    # -- training ---------------------------------------------------------
    def train_step(self, x: np.ndarray, mask: np.ndarray, actions: np.ndarray, returns: np.ndarray,
                   lr: float, value_coef: float = 0.5, entropy_coef: float = 0.01,
                   max_grad_norm: float = 1.0, supervised: bool = False,
                   adv_clip: float = 3.0) -> dict[str, float]:
        """One policy-gradient (actor-critic) Adam step. Returns loss statistics.

        With ``supervised=True`` the advantage is fixed to 1, which turns the
        policy loss into plain cross-entropy (imitation of the given actions).
        """
        xt = torch.as_tensor(x, dtype=torch.float32)
        mt = torch.as_tensor(mask, dtype=torch.bool)
        at = torch.as_tensor(actions, dtype=torch.long)
        rt = torch.as_tensor(returns, dtype=torch.float32)

        self.train()
        logits, value = self(xt)
        logp_all = torch.log_softmax(logits.masked_fill(~mt, NEG_INF), dim=-1)
        probs = logp_all.exp() * mt
        logp_all = torch.where(mt, logp_all, torch.zeros_like(logp_all))

        if supervised:
            adv = torch.ones_like(rt)
        else:
            adv = (rt - value).detach().clamp(-adv_clip, adv_clip)

        logp = logp_all.gather(1, at[:, None])[:, 0]
        entropy = -(probs * logp_all).sum(dim=1)
        policy_loss = -(adv * logp).mean()
        value_loss = ((value - rt) ** 2).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
        self._optimize(loss, lr, max_grad_norm)
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                "entropy": entropy.mean().item()}

    def q_step(self, x: np.ndarray, actions: np.ndarray, returns: np.ndarray, lr: float,
               max_grad_norm: float = 1.0) -> dict[str, float]:
        """Action-value regression: the action head predicts the return Q(s, a).

        Only the output of the action that was actually taken is trained
        (a contextual bandit). Used for bidding, which is a one-shot decision.
        """
        xt = torch.as_tensor(x, dtype=torch.float32)
        at = torch.as_tensor(actions, dtype=torch.long)
        rt = torch.as_tensor(returns, dtype=torch.float32)
        self.train()
        q, _ = self(xt)
        taken = q.gather(1, at[:, None])[:, 0]
        loss = ((taken - rt) ** 2).mean()
        self._optimize(loss, lr, max_grad_norm)
        return {"policy_loss": 0.0, "value_loss": loss.item(), "entropy": 0.0}

    def _optimize(self, loss: torch.Tensor, lr: float, max_grad_norm: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()
        self.steps += 1

    # -- persistence ------------------------------------------------------
    def _named_arrays(self) -> dict[str, nn.Parameter]:
        """Parameters under the legacy names; weights are (in, out) in files."""
        named: dict[str, nn.Parameter] = {}
        for k, layer in enumerate(self.trunk):
            named[f"W{k}"], named[f"b{k}"] = layer.weight, layer.bias
        named["Wp"], named["bp"] = self.policy_head.weight, self.policy_head.bias
        named["Wv"], named["bv"] = self.value_head.weight, self.value_head.bias
        return named

    @staticmethod
    def _to_file(name: str, t: torch.Tensor) -> np.ndarray:
        a = t.detach().numpy()
        return np.ascontiguousarray(a.T if name.startswith("W") else a)

    @staticmethod
    def _from_file(name: str, a: np.ndarray) -> torch.Tensor:
        a = np.asarray(a, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(a.T if name.startswith("W") else a))

    def state_dict_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, p in self._named_arrays().items():
            state = self.optimizer.state.get(p, {})
            out[f"{prefix}.{name}"] = self._to_file(name, p)
            out[f"{prefix}.adam_m.{name}"] = self._to_file(name, state["exp_avg"]) if state else np.zeros_like(self._to_file(name, p))
            out[f"{prefix}.adam_v.{name}"] = self._to_file(name, state["exp_avg_sq"]) if state else np.zeros_like(self._to_file(name, p))
        out[f"{prefix}.shape"] = np.array([self.input_dim, self.num_actions, *self.hidden], dtype=np.int64)
        out[f"{prefix}.steps"] = np.array(self.steps, dtype=np.int64)
        return out

    @classmethod
    def from_state_dict(cls, data, prefix: str) -> PolicyValueNet:
        shape = [int(s) for s in data[f"{prefix}.shape"]]
        net = cls(shape[0], shape[1], shape[2:])
        net.steps = int(data[f"{prefix}.steps"])
        with torch.no_grad():
            for name, p in net._named_arrays().items():
                p.copy_(cls._from_file(name, data[f"{prefix}.{name}"]))
                if net.steps:
                    net.optimizer.state[p] = {
                        "step": torch.tensor(float(net.steps)),
                        "exp_avg": cls._from_file(name, data[f"{prefix}.adam_m.{name}"]),
                        "exp_avg_sq": cls._from_file(name, data[f"{prefix}.adam_v.{name}"]),
                    }
        return net
