import random

import numpy as np

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


def test_gradient_matches_numeric():
    net = PolicyValueNet(6, 4, (5,), seed=1)
    x = np.random.default_rng(0).standard_normal((3, 6)).astype(np.float64)
    mask = np.array([[1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool)
    actions = np.array([0, 2, 1])
    for k in net.params:
        net.params[k] = net.params[k].astype(np.float64)

    def loss() -> float:
        acts, logits, value = net._forward(x)
        p = net.masked_softmax(logits, mask)
        logp = np.log(p[np.arange(3), actions])
        return float(-logp.mean())

    # analytic gradient with supervised=True, no value/entropy terms and plain SGD
    captured = {}
    net._adam = lambda grads, lr: captured.update(grads)
    net.train_step(x, mask, actions, np.zeros(3), lr=0.0, value_coef=0.0, entropy_coef=0.0,
                   max_grad_norm=1e9, supervised=True)
    eps = 1e-6
    w = net.params["W0"]
    numeric = np.zeros_like(w)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            old = w[i, j]
            w[i, j] = old + eps
            up = loss()
            w[i, j] = old - eps
            down = loss()
            w[i, j] = old
            numeric[i, j] = (up - down) / (2 * eps)
    assert np.allclose(captured["W0"], numeric, atol=1e-5)


def test_model_save_and_load(tmp_path):
    model = ModelBundle.fresh(TierConfig("t", hidden=(8, 4)), seed=0)
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = ModelBundle.load(path)
    for k, v in model.play_net.params.items():
        assert np.array_equal(v, loaded.play_net.params[k])


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
