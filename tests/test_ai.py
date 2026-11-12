import json
import random

import numpy as np
import torch

from schafskopf.ai.agent import ModelBundle, RLPlayer
from schafskopf.ai.config import TierConfig, load_tier
from schafskopf.ai.encoding import BID_DIM, BID_LAYOUT, PLAY_DIM, PLAY_LAYOUT, encode_bid, encode_play
from schafskopf.ai.learner import Learner, LossConfig
from schafskopf.ai.legacy import load_legacy_npz
from schafskopf.ai.network import PolicyValueNet, masked_logits
from schafskopf.ai.training import Trainer, trajectories_to_arrays
from schafskopf.game import Round


def test_layout_sizes():
    assert PLAY_DIM == sum(n for _, n in PLAY_LAYOUT)
    assert BID_DIM == sum(n for _, n in BID_LAYOUT)


def test_encodings_in_real_round():
    rng = random.Random(3)
    cfg = TierConfig("test", hidden=(16,))
    model = ModelBundle.fresh(cfg, seed=0)
    records = []
    players = [RLPlayer(f"p{s}", model, 1.0, rng=rng, recorder=records) for s in range(4)]
    Round(players, 0, rng=rng).run()
    arrays = trajectories_to_arrays(records)
    assert arrays["bid_states"].shape == (4, BID_DIM)
    assert arrays["play_states"].shape[1] == PLAY_DIM
    # every recorded action is legal
    assert arrays["play_masks"][np.arange(len(arrays["play_actions"])), arrays["play_actions"]].all()


def test_gradient_matches_numeric():
    net = PolicyValueNet(6, 4, (5,), seed=1).double()
    x = np.random.default_rng(0).standard_normal((3, 6))
    mask = np.array([[1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool)
    actions = np.array([0, 2, 1])
    w = net.trunk[0].weight

    def loss() -> float:
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(x))
        p = torch.softmax(masked_logits(logits, torch.from_numpy(mask)), dim=-1).numpy()
        return float(-np.log(p[np.arange(3), actions]).mean())

    # autograd gradient: supervised, no value/entropy terms, no clipping, lr=0 (no update)
    learner = Learner(net, lr=0.0, loss=LossConfig(value_coef=0.0, entropy_coef=0.0, max_grad_norm=1e9))
    learner.train_step(x, mask, actions, np.zeros(3), supervised=True)
    analytic = w.grad.numpy().copy()
    eps = 1e-6
    numeric = np.zeros_like(analytic)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            old = w.data[i, j].item()
            w.data[i, j] = old + eps
            up = loss()
            w.data[i, j] = old - eps
            down = loss()
            w.data[i, j] = old
            numeric[i, j] = (up - down) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_model_save_and_load(tmp_path):
    model = ModelBundle.fresh(TierConfig("t", hidden=(8, 4)), seed=0)
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = ModelBundle.load(path)
    for k, v in model.play_net.state_dict().items():
        assert torch.equal(v, loaded.play_net.state_dict()[k])


def test_legacy_npz_import(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {"meta": np.array(json.dumps({"episodes": 3}))}
    for prefix, (d_in, n_out) in {"bid": (BID_DIM, 4), "play": (PLAY_DIM, 32)}.items():
        sizes = (d_in, 8)
        arrays[f"{prefix}.shape"] = np.array([d_in, n_out, 8])
        arrays[f"{prefix}.W0"] = rng.standard_normal((d_in, 8)).astype(np.float32)
        arrays[f"{prefix}.b0"] = rng.standard_normal(8).astype(np.float32)
        arrays[f"{prefix}.Wp"] = rng.standard_normal((8, n_out)).astype(np.float32)
        arrays[f"{prefix}.bp"] = rng.standard_normal(n_out).astype(np.float32)
        arrays[f"{prefix}.Wv"] = rng.standard_normal((8, 1)).astype(np.float32)
        arrays[f"{prefix}.bv"] = rng.standard_normal(1).astype(np.float32)
    np.savez(tmp_path / "old.npz", **arrays)
    bundle = load_legacy_npz(tmp_path / "old.npz")
    x = rng.standard_normal(PLAY_DIM).astype(np.float32)
    h = np.maximum(x @ arrays["play.W0"] + arrays["play.b0"], 0)
    expected = h @ arrays["play.Wp"] + arrays["play.bp"]
    assert np.allclose(bundle.play_net.logits(x), expected, atol=1e-4)
    assert bundle.meta["episodes"] == 3


def test_short_training_run(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHAFSKOPF_HOME", str(tmp_path))
    cfg = TierConfig("tiny", hidden=(16,), batch_episodes=10, imitation_episodes=10, imitation_epochs=1)
    trainer = Trainer(cfg, seed=0)
    trainer.train(20)
    assert (tmp_path / "models" / "tiny.pt").exists()
    assert len(list((tmp_path / "training_data" / "tiny").glob("batch_*.npz"))) == 2
    assert (tmp_path / "training_data" / "tiny" / "log.csv").exists()
    files, _ = Trainer(cfg, seed=0).replay(epochs=1)
    assert files == 2


def test_tier_config_loads():
    for tier in ("easy", "medium", "hard"):
        assert load_tier(tier).hidden
