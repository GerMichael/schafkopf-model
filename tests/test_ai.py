import random

import numpy as np
import torch

from schafskopf.ai.agent import ModelBundle, RLPlayer
from schafskopf.ai.config import TierConfig, load_tier
from schafskopf.ai.encoding import BID_DIM, BID_LAYOUT, PLAY_DIM, PLAY_LAYOUT, encode_bid, encode_play
from schafskopf.ai.network import PolicyValueNet
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


def test_supervised_step_learns_and_respects_mask():
    net = PolicyValueNet(6, 4, (16,), seed=1)
    x = np.random.default_rng(0).standard_normal((8, 6)).astype(np.float32)
    mask = np.ones((8, 4), dtype=bool)
    mask[:, 3] = False
    actions = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    first = net.train_step(x, mask, actions, np.zeros(8), lr=1e-2, supervised=True, entropy_coef=0.0)
    for _ in range(200):
        last = net.train_step(x, mask, actions, np.zeros(8), lr=1e-2, supervised=True, entropy_coef=0.0)
    assert last["policy_loss"] < first["policy_loss"]
    probs, _ = net.predict(x[0], mask[0])
    assert probs[3] == 0 and abs(probs.sum() - 1) < 1e-5


def test_q_step_regresses_taken_action():
    net = PolicyValueNet(4, 3, (16,), seed=2)
    x = np.random.default_rng(1).standard_normal((16, 4)).astype(np.float32)
    actions = np.arange(16) % 3
    returns = np.where(actions == 0, 1.0, -1.0).astype(np.float32)
    for _ in range(300):
        stats = net.q_step(x, actions, returns, lr=1e-2)
    assert stats["value_loss"] < 0.05


def test_model_save_and_load(tmp_path):
    model = ModelBundle.fresh(TierConfig("t", hidden=(8, 4)), seed=0)
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = ModelBundle.load(path)
    for k, v in model.play_net.state_dict().items():
        assert torch.equal(v, loaded.play_net.state_dict()[k])


def test_short_training_run(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHAFSKOPF_HOME", str(tmp_path))
    cfg = TierConfig("tiny", hidden=(16,), batch_episodes=10, imitation_episodes=10, imitation_epochs=1)
    trainer = Trainer(cfg, seed=0)
    trainer.train(20)
    assert (tmp_path / "models" / "tiny.npz").exists()
    assert len(list((tmp_path / "training_data" / "tiny").glob("batch_*.npz"))) == 2
    assert (tmp_path / "training_data" / "tiny" / "log.csv").exists()
    files, _ = Trainer(cfg, seed=0).replay(epochs=1)
    assert files == 2


def test_tier_config_loads():
    for tier in ("easy", "medium", "hard"):
        assert load_tier(tier).hidden
