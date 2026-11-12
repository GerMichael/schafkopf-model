"""Import of the old pure-numpy checkpoints (``models/<tier>.npz``) into the torch format.

Only weights are converted; Adam moments are dropped (the optimizer restarts fresh).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .network import PolicyValueNet


def _load_net(data, prefix: str) -> PolicyValueNet:
    shape = [int(s) for s in data[f"{prefix}.shape"]]
    net = PolicyValueNet(shape[0], shape[1], shape[2:])
    # old weights were stored as (in, out); torch wants (out, in)
    pairs = {"policy_head": "p", "value_head": "v", **{f"trunk.{2 * k}": str(k) for k in range(len(net.hidden))}}
    params = dict(net.named_parameters())
    with torch.no_grad():
        for torch_name, old in pairs.items():
            params[f"{torch_name}.weight"].copy_(torch.from_numpy(data[f"{prefix}.W{old}"].T.copy()))
            params[f"{torch_name}.bias"].copy_(torch.from_numpy(data[f"{prefix}.b{old}"].copy()))
    return net


def load_legacy_npz(path: Path):
    """Read an old ``.npz`` checkpoint into a ``ModelBundle``."""
    from .agent import ModelBundle

    with np.load(path) as data:
        return ModelBundle(_load_net(data, "bid"), _load_net(data, "play"), json.loads(str(data["meta"])))
