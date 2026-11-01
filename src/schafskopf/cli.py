"""Command-line entry point: interactive game menu and training commands.

    uv run schafskopf                      # interactive menu
    uv run schafskopf play                 # same
    uv run schafskopf train --tier hard    # continue training a tier
    uv run schafskopf train-all            # train easy, medium and hard
    uv run schafskopf replay --tier hard   # re-learn from stored vectors
    uv run schafskopf evaluate --tier hard # benchmark against the rule-based bot
    uv run schafskopf data                 # show models and training data
    uv run schafskopf rules                # print the rules
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

from .ai.config import TIERS, data_dir, load_tier, model_path, project_root
from .game import GameSettings, Match
from .players import HeuristicPlayer, Player, RandomPlayer
from .players.human import HumanPlayer
from .rules import GameType
from .ui import QuitGame, ask, ask_choice, ask_int, bold, euro

DEFAULT_NAMES = ["Sepp", "Resi", "Wast", "Vroni"]

MODES = [
    ("Standard – Sauspiel, Wenz and Solo; Ramsch when everybody passes",
     GameSettings()),
    ("Standard without Ramsch – cards are thrown in when everybody passes",
     GameSettings(ramsch_if_all_pass=False)),
    ("Sauspiel only (beginner) – Ramsch when everybody passes",
     GameSettings(allowed=frozenset({GameType.SAUSPIEL}))),
    ("Ramsch only – nobody may announce, the most points lose",
     GameSettings(allowed=frozenset())),
    ("Custom – choose the allowed games yourself", None),
]

PLAYER_TYPES = ["Human", "AI – easy", "AI – medium", "AI – hard", "Rule-based bot", "Random bot"]


# ---------------------------------------------------------------- players --
_model_cache: dict = {}


def make_ai(tier: str, name: str, rng: random.Random) -> Player:
    from .ai.agent import ModelBundle, RLPlayer

    path = model_path(tier)
    if not path.exists():
        print(f"  ! No trained model for '{tier}' at {path}.")
        print(f"    Train it with:  uv run schafskopf train --tier {tier}")
        print("    Using the rule-based bot instead for now.")
        return HeuristicPlayer(name, rng)
    if tier not in _model_cache:
        _model_cache[tier] = ModelBundle.load(path)
    cfg = load_tier(tier)
    return RLPlayer(name, _model_cache[tier], cfg.temperature, cfg.epsilon, rng, tier=tier)


def make_player(kind: int, name: str, rng: random.Random, hot_seat: bool) -> Player:
    match kind:
        case 0:
            return HumanPlayer(name, hot_seat=hot_seat)
        case 1 | 2 | 3:
            return make_ai(TIERS[kind - 1], name, rng)
        case 4:
            return HeuristicPlayer(name, rng)
        case _:
            return RandomPlayer(name, rng)


# --------------------------------------------------------------- new game --
def choose_settings() -> GameSettings:
    print(bold("\nChoose a game mode:"))
    choice = ask_choice("Mode:", [label for label, _ in MODES], default=0)
    settings = MODES[choice][1]
    if settings is not None:
        return settings
    allowed = set()
    for game in (GameType.SAUSPIEL, GameType.WENZ, GameType.SOLO):
        if ask(f"  Allow {game.label}? [Y/n] ").lower() not in {"n", "no"}:
            allowed.add(game)
    ramsch = ask("  Play Ramsch when everybody passes? [Y/n] ").lower() not in {"n", "no"}
    return GameSettings(allowed=frozenset(allowed), ramsch_if_all_pass=ramsch)


def choose_players(rng: random.Random) -> list[Player]:
    print(bold("\nWho sits at the table? (seats play in clockwise order)"))
    kinds, names = [], []
    for seat in range(4):
        print(f"\n Seat {seat + 1}:")
        kind = ask_choice("  Player type:", PLAYER_TYPES, default=0 if seat == 0 else 3)
        name = ask(f"  Name [{DEFAULT_NAMES[seat]}]: ") or DEFAULT_NAMES[seat]
        kinds.append(kind)
        names.append(name)
    hot_seat = kinds.count(0) > 1
    return [make_player(k, n, rng, hot_seat) for k, n in zip(kinds, names)]


def new_game() -> None:
    from .ui import ConsoleObserver

    rng = random.Random()
    settings = choose_settings()
    rounds = ask_int("\nNumber of rounds", 8, 1, 1000)
    players = choose_players(rng)
    humans = sum(p.is_human for p in players)
    delay = 0.4 if humans else 0.0
    if not humans:
        delay = float(ask("  Pause between moves in seconds for watching [0]: ") or 0)
    observer = ConsoleObserver(delay=delay)
    match = Match(players, settings, rounds, rng, observers=[observer])

    print(bold("\nThe table:"))
    for seat, p in enumerate(players):
        print(f"  Seat {seat + 1}: {p.name} ({p.kind})")
    try:
        for n in range(rounds):
            match.play_round()
            if humans and n < rounds - 1:
                ask("\n  Press Enter for the next round (q to stop) ")
    except QuitGame:
        print("\n  Game stopped.")
    show_standings(match)


def show_standings(match: Match) -> None:
    print(bold("\n═══ Final standings ═══"))
    order = sorted(range(4), key=lambda s: match.totals[s], reverse=True)
    for rank, seat in enumerate(order, 1):
        print(f"  {rank}. {match.players[seat].name:<12} {euro(match.totals[seat])}")
    if match.history:
        winners = ", ".join(match.players[s].name for s in match.winners())
        print(bold(f"\n  Winner: {winners}"))


# --------------------------------------------------------------- training --
def print_batch(stats) -> None:
    print(f"  batch {stats.batch:>4} | rounds {stats.episodes:>7} | reward {stats.mean_reward:+.3f} | "
          f"payout {stats.mean_payout:+6.1f} ct | declares {stats.declare_rate:5.1%} "
          f"(won {stats.declarer_win_rate:5.1%}) | entropy {stats.entropy:.3f} | {stats.seconds:.1f}s")


def train_tier(tier: str, episodes: int | None = None, reset: bool = False, opponents: str | None = None,
               learning_rate: float | None = None, save_data: bool = True, seed: int | None = None,
               imitation: int | None = None, evaluate_rounds: int = 500) -> None:
    from .ai.training import Trainer, evaluate

    cfg = load_tier(tier)
    if imitation is not None:
        cfg = replace(cfg, imitation_episodes=imitation)
    trainer = Trainer(cfg, reset=reset, seed=seed, opponents=opponents,
                      learning_rate=learning_rate, save_data=save_data)
    total = episodes or cfg.episodes
    print(bold(f"\nTraining tier '{tier}' for {total} rounds "
               f"(opponents: {trainer.opponents}, hidden {list(cfg.hidden)})"))
    print(f"  model: {trainer.model_file}\n  data:  {trainer.data_dir}")

    def on_imitation(samples: int, losses: dict) -> None:
        print(f"  imitation warm start: {cfg.imitation_episodes} rounds of the rule-based bot, "
              f"{samples} seat-rounds, policy loss {losses['policy_loss']:.3f}")

    trainer.train(total, progress=print_batch, on_imitation=on_imitation)
    if evaluate_rounds:
        result = evaluate(trainer.model, evaluate_rounds, seed=seed)
        print(f"  evaluation vs rule-based bot over {evaluate_rounds} rounds: "
              f"{result['model_cents_per_round']:+.2f} ct per round and seat")


def show_data() -> None:
    import numpy as np

    from .ai.agent import ModelBundle

    print(bold(f"\nProject folder: {project_root()}"))
    for tier in TIERS:
        path = model_path(tier)
        print(bold(f"\n[{tier}]"))
        if path.exists():
            meta = ModelBundle.load(path).meta
            print(f"  model: {path.name}, rounds trained {meta.get('episodes', 0)}, "
                  f"imitation {meta.get('imitation_episodes', 0)}, batches {meta.get('batches', 0)}, "
                  f"hidden {meta.get('hidden')}, updated {meta.get('updated', '?')}")
        else:
            print("  model: not trained yet")
        folder = data_dir(tier)
        files = sorted(folder.glob("*.npz")) if folder.exists() else []
        bid = play = 0
        for f in files:
            with np.load(f) as d:
                bid += len(d["bid_actions"])
                play += len(d["play_actions"])
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"  data:  {len(files)} vector files ({size:.1f} MB), {bid} bid vectors, {play} card-play vectors")


def training_menu() -> None:
    print(bold("\nTrain AI opponents"))
    tier = TIERS[ask_choice("Tier:", [f"{t} ({load_tier(t).episodes} rounds by default)" for t in TIERS], 1)]
    action = ask_choice("What to do:", [
        "Continue training with self-play (keeps existing knowledge)",
        "Start over (reset the model) and train",
        "Re-learn from the stored training vectors only",
        "Evaluate against the rule-based bot",
    ], 0)
    if action in (0, 1):
        episodes = ask_int("Rounds to play", load_tier(tier).episodes, 1, 10_000_000)
        train_tier(tier, episodes, reset=action == 1)
    elif action == 2:
        epochs = ask_int("Passes over the data", 1, 1, 100)
        replay_tier(tier, epochs)
    else:
        evaluate_tier(tier, ask_int("Rounds", 1000, 10, 1_000_000), "heuristic")


def replay_tier(tier: str, epochs: int, last: int | None = None) -> None:
    from .ai.training import Trainer

    files, losses = Trainer(load_tier(tier)).replay(epochs, last)
    if not files:
        print(f"  No stored vectors in {data_dir(tier)} – run `train` first.")
        return
    print(f"  Re-learned from {files} files: policy loss {losses['policy_loss']:.3f}, "
          f"value loss {losses['value_loss']:.3f}")


def evaluate_tier(tier: str, rounds: int, against: str, seed: int | None = None) -> None:
    from .ai.agent import ModelBundle
    from .ai.training import evaluate

    if not model_path(tier).exists():
        print(f"  No model for '{tier}' yet.")
        return
    model = ModelBundle.load(model_path(tier))
    opponent, mode, opp_temp, opp_eps = None, against, 0.0, 0.0
    if against in TIERS:
        if not model_path(against).exists():
            print(f"  No model for '{against}' yet.")
            return
        opp_cfg = load_tier(against)
        opponent, mode = ModelBundle.load(model_path(against)), "model"
        opp_temp, opp_eps = opp_cfg.temperature, opp_cfg.epsilon
    cfg = load_tier(tier)
    r = evaluate(model, rounds, mode, seed, opponent, cfg.temperature, cfg.epsilon, opp_temp, opp_eps)
    print(f"  {tier} vs {against} over {rounds} rounds:")
    print(f"    {tier:>9}: {r['model_cents_per_round']:+.2f} ct per round and seat")
    print(f"    {against:>9}: {r['opponent_cents_per_round']:+.2f} ct per round and seat")
    print(f"    {tier} announced a game in {r['model_declare_rate']:.1%} of its seat-rounds and won "
          f"{r['model_declarer_win_rate']:.1%} of them")


def show_rules() -> None:
    rules = project_root() / "docs" / "rules.md"
    print(rules.read_text() if rules.exists() else "docs/rules.md not found.")


def main_menu() -> None:
    print(bold("\n♣ ♠ ♥ ♦  Schafkopf  ♦ ♥ ♠ ♣"))
    print("  (type 'q' at any prompt to go back / quit)")
    while True:
        print(bold("\nMain menu"))
        try:
            choice = ask_choice("Choice:", ["Start a new game", "Train AI opponents",
                                            "Show models and training data", "Show the rules", "Quit"], 0)
            if choice == 0:
                new_game()
            elif choice == 1:
                training_menu()
            elif choice == 2:
                show_data()
            elif choice == 3:
                show_rules()
            else:
                return
        except QuitGame:
            if choice == 4 or ask_quit():
                return


def ask_quit() -> bool:
    try:
        return ask("\nQuit Schafkopf? [y/N] ").lower() in {"y", "yes", "j", "ja"}
    except QuitGame:
        return True


# -------------------------------------------------------------------- cli --
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="schafskopf", description="Schafkopf with reinforcement-learning AI")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("play", help="interactive menu (default)")

    t = sub.add_parser("train", help="train one AI tier with self-play")
    t.add_argument("--tier", choices=TIERS, required=True)
    t.add_argument("--episodes", type=int, help="rounds to play (default: from config/tiers.toml)")
    t.add_argument("--reset", action="store_true", help="discard the existing model and start fresh")
    t.add_argument("--opponents", choices=["self", "heuristic", "random", "mixed"])
    t.add_argument("--lr", type=float, help="override the learning rate")
    t.add_argument("--imitation", type=int, help="imitation warm-start rounds for a fresh model")
    t.add_argument("--no-save-data", action="store_true", help="do not write vector files")
    t.add_argument("--seed", type=int)
    t.add_argument("--eval-rounds", type=int, default=500)

    ta = sub.add_parser("train-all", help="train easy, medium and hard")
    ta.add_argument("--reset", action="store_true")
    ta.add_argument("--scale", type=float, default=1.0, help="multiply the configured rounds")
    ta.add_argument("--seed", type=int)

    r = sub.add_parser("replay", help="re-learn from the stored training vectors")
    r.add_argument("--tier", choices=TIERS, required=True)
    r.add_argument("--epochs", type=int, default=1)
    r.add_argument("--last", type=int, help="only use the newest N batch files")

    e = sub.add_parser("evaluate", help="benchmark a tier")
    e.add_argument("--tier", choices=TIERS, required=True)
    e.add_argument("--rounds", type=int, default=1000)
    e.add_argument("--against", choices=["heuristic", "random", *TIERS], default="heuristic")
    e.add_argument("--seed", type=int)

    sub.add_parser("data", help="show models and stored training data")
    sub.add_parser("rules", help="print the rules")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.command:
            case None | "play":
                main_menu()
            case "train":
                train_tier(args.tier, args.episodes, args.reset, args.opponents, args.lr,
                           not args.no_save_data, args.seed, args.imitation, args.eval_rounds)
            case "train-all":
                for tier in TIERS:
                    episodes = max(1, int(load_tier(tier).episodes * args.scale))
                    train_tier(tier, episodes, args.reset, seed=args.seed)
            case "replay":
                replay_tier(args.tier, args.epochs, args.last)
            case "evaluate":
                evaluate_tier(args.tier, args.rounds, args.against, args.seed)
            case "data":
                show_data()
            case "rules":
                show_rules()
    except KeyboardInterrupt:
        print("\n  Interrupted – the model saved after the last completed batch is kept.")
        return 130
    except QuitGame:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
