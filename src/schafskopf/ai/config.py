"""Project paths and the hand-editable tier configuration (config/tiers.toml)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

TIERS = ("easy", "medium", "hard")


def project_root() -> Path:
    """Folder that holds config/, models/ and training_data/.

    Override with the ``SCHAFSKOPF_HOME`` environment variable.
    """
    if env := os.environ.get("SCHAFSKOPF_HOME"):
        return Path(env).expanduser().resolve()
    here = Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "tiers.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[3]


def config_path() -> Path:
    return project_root() / "config" / "tiers.toml"


def model_path(tier: str) -> Path:
    return project_root() / "models" / f"{tier}.pt"


def data_dir(tier: str) -> Path:
    return project_root() / "training_data" / tier


@dataclass(frozen=True)
class TierConfig:
    name: str
    hidden: tuple[int, ...] = (128, 64)
    episodes: int = 10_000
    batch_episodes: int = 250
    learning_rate: float = 0.0007
    epochs: int = 3
    minibatch: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    reward_scale: float = 50.0
    shaping: float = 0.3
    opponents: str = "mixed"
    learner_seats: int = 2
    temperature: float = 0.5
    epsilon: float = 0.0
    keep_batches: int = 40
    bid_epsilon: float = 0.1
    imitation_episodes: int = 2000
    imitation_epochs: int = 5
    imitation_bid_explore: float = 0.3


def load_tier(name: str, path: Path | None = None) -> TierConfig:
    path = path or config_path()
    raw: dict = {}
    if path.is_file():
        with path.open("rb") as f:
            raw = tomllib.load(f).get(name, {})
    known = {f.name for f in fields(TierConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in [{name}] of {path}: {', '.join(sorted(unknown))}")
    if "hidden" in raw:
        raw["hidden"] = tuple(raw["hidden"])
    if raw.get("opponents", "mixed") not in {"self", "heuristic", "random", "mixed"}:
        raise ValueError(f"[{name}].opponents must be self, heuristic, random or mixed")
    return TierConfig(name=name, **raw)
