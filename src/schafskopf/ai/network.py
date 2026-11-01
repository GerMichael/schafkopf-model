"""A small policy/value neural network in pure numpy, trained with Adam.

Architecture (shared trunk, two heads)::

    x ─ Dense ─ ReLU ─ Dense ─ ReLU ─┬─ Dense → action logits (masked softmax)
                                     └─ Dense → state value V(s)

Card play (actor-critic) – loss for one sample with return R, chosen action a
and advantage A = clip(R - V(s))::

    L = -A · log π(a|s)  +  c_v · (V(s) - R)²  -  c_e · H(π(·|s))

Bidding (contextual bandit) – the action head predicts Q(s, a)::

    L = (Q(s, a) - R)²
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NEG_INF = -1e9


class PolicyValueNet:
    def __init__(self, input_dim: int, num_actions: int, hidden: Sequence[int] = (128, 64),
                 seed: int | None = None) -> None:
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden = tuple(int(h) for h in hidden)
        rng = np.random.default_rng(seed)
        sizes = (input_dim, *self.hidden)
        self.params: dict[str, np.ndarray] = {}
        for k in range(len(self.hidden)):
            self.params[f"W{k}"] = (rng.standard_normal((sizes[k], sizes[k + 1])) * np.sqrt(2 / sizes[k])).astype(np.float32)
            self.params[f"b{k}"] = np.zeros(sizes[k + 1], dtype=np.float32)
        last = sizes[-1]
        self.params["Wp"] = (rng.standard_normal((last, num_actions)) * 0.01).astype(np.float32)
        self.params["bp"] = np.zeros(num_actions, dtype=np.float32)
        self.params["Wv"] = (rng.standard_normal((last, 1)) * 0.01).astype(np.float32)
        self.params["bv"] = np.zeros(1, dtype=np.float32)
        self._adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.steps = 0

    # -- inference --------------------------------------------------------
    def _forward(self, x: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        activations = [x]
        h = x
        for k in range(len(self.hidden)):
            h = np.maximum(h @ self.params[f"W{k}"] + self.params[f"b{k}"], 0.0)
            activations.append(h)
        logits = h @ self.params["Wp"] + self.params["bp"]
        value = (h @ self.params["Wv"] + self.params["bv"])[:, 0]
        return activations, logits, value

    @staticmethod
    def masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        z = np.where(mask, logits / max(temperature, 1e-6), NEG_INF)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z) * mask
        return e / e.sum(axis=-1, keepdims=True)

    def predict(self, x: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> tuple[np.ndarray, float]:
        """Action probabilities and value for a single state vector."""
        _, logits, value = self._forward(x[None, :])
        return self.masked_softmax(logits, mask[None, :], temperature)[0], float(value[0])

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x[None, :])[1][0]

    # -- training ---------------------------------------------------------
    def train_step(self, x: np.ndarray, mask: np.ndarray, actions: np.ndarray, returns: np.ndarray,
                   lr: float, value_coef: float = 0.5, entropy_coef: float = 0.01,
                   max_grad_norm: float = 1.0, supervised: bool = False,
                   adv_clip: float = 3.0) -> dict[str, float]:
        """One policy-gradient (actor-critic) Adam step. Returns loss statistics.

        With ``supervised=True`` the advantage is fixed to 1, which turns the
        policy loss into plain cross-entropy (imitation of the given actions).
        """
        n = len(x)
        acts, logits, value = self._forward(x)
        probs = self.masked_softmax(logits, mask)
        logp = np.log(np.where(mask, probs, 1.0) + 1e-12)

        if supervised:
            adv = np.ones(n, dtype=np.float32)
        else:
            adv = np.clip(returns - value, -adv_clip, adv_clip)

        onehot = np.zeros_like(probs)
        onehot[np.arange(n), actions] = 1.0
        entropy = -(probs * logp).sum(axis=1)

        # d/dlogits of  -A·log π(a)  is  A·(π - onehot);  of  -c_e·H  is  c_e·π·(log π + H)
        d_logits = adv[:, None] * (probs - onehot)
        d_logits += entropy_coef * probs * (logp + entropy[:, None])
        d_logits = np.where(mask, d_logits, 0.0) / n
        d_value = (2 * value_coef * (value - returns) / n)[:, None]
        self._apply(self._backward(acts, d_logits, d_value), lr, max_grad_norm)

        policy_loss = float(-(adv * logp[np.arange(n), actions]).mean())
        value_loss = float(((value - returns) ** 2).mean())
        return {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": float(entropy.mean())}

    def q_step(self, x: np.ndarray, actions: np.ndarray, returns: np.ndarray, lr: float,
               max_grad_norm: float = 1.0) -> dict[str, float]:
        """Action-value regression: the action head predicts the return Q(s, a).

        Only the output of the action that was actually taken is trained
        (a contextual bandit). Used for bidding, which is a one-shot decision.
        """
        n = len(x)
        acts, q, _ = self._forward(x)
        taken = q[np.arange(n), actions]
        d_logits = np.zeros_like(q)
        d_logits[np.arange(n), actions] = 2 * (taken - returns) / n
        self._apply(self._backward(acts, d_logits, np.zeros((n, 1), dtype=q.dtype)), lr, max_grad_norm)
        return {"policy_loss": 0.0, "value_loss": float(((taken - returns) ** 2).mean()), "entropy": 0.0}

    def _backward(self, acts: list[np.ndarray], d_logits: np.ndarray, d_value: np.ndarray) -> dict[str, np.ndarray]:
        h = acts[-1]
        grads = {
            "Wp": h.T @ d_logits, "bp": d_logits.sum(0),
            "Wv": h.T @ d_value, "bv": d_value.sum(0),
        }
        dh = d_logits @ self.params["Wp"].T + d_value @ self.params["Wv"].T
        for k in reversed(range(len(self.hidden))):
            dh = dh * (acts[k + 1] > 0)
            grads[f"W{k}"] = acts[k].T @ dh
            grads[f"b{k}"] = dh.sum(0)
            if k:
                dh = dh @ self.params[f"W{k}"].T
        return grads

    def _apply(self, grads: dict[str, np.ndarray], lr: float, max_grad_norm: float) -> None:
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        if norm > max_grad_norm:
            grads = {k: g * (max_grad_norm / norm) for k, g in grads.items()}
        self._adam(grads, lr)

    def _adam(self, grads: dict[str, np.ndarray], lr: float, b1: float = 0.9, b2: float = 0.999) -> None:
        self.steps += 1
        for k, g in grads.items():
            self._adam_m[k] = b1 * self._adam_m[k] + (1 - b1) * g
            self._adam_v[k] = b2 * self._adam_v[k] + (1 - b2) * g * g
            m_hat = self._adam_m[k] / (1 - b1 ** self.steps)
            v_hat = self._adam_v[k] / (1 - b2 ** self.steps)
            self.params[k] -= (lr * m_hat / (np.sqrt(v_hat) + 1e-8)).astype(np.float32)

    # -- persistence ------------------------------------------------------
    def state_dict(self, prefix: str) -> dict[str, np.ndarray]:
        out = {f"{prefix}.{k}": v for k, v in self.params.items()}
        out.update({f"{prefix}.adam_m.{k}": v for k, v in self._adam_m.items()})
        out.update({f"{prefix}.adam_v.{k}": v for k, v in self._adam_v.items()})
        out[f"{prefix}.shape"] = np.array([self.input_dim, self.num_actions, *self.hidden], dtype=np.int64)
        out[f"{prefix}.steps"] = np.array(self.steps, dtype=np.int64)
        return out

    @classmethod
    def from_state_dict(cls, data, prefix: str) -> PolicyValueNet:
        shape = [int(s) for s in data[f"{prefix}.shape"]]
        net = cls(shape[0], shape[1], shape[2:])
        for k in net.params:
            net.params[k] = data[f"{prefix}.{k}"].astype(np.float32)
            net._adam_m[k] = data[f"{prefix}.adam_m.{k}"].astype(np.float32)
            net._adam_v[k] = data[f"{prefix}.adam_v.{k}"].astype(np.float32)
        net.steps = int(data[f"{prefix}.steps"])
        return net
