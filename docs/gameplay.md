# Gameplay – Playing in the Terminal

## 1. Installation

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.14 (uv downloads it if needed).

```bash
cd schafskopf
uv sync            # creates .venv with Python 3.14 and installs numpy and PyTorch
uv run schafskopf  # starts the interactive menu
```

## 2. Main menu

```
Main menu
  [1] Start a new game (default)
  [2] Train AI opponents
  [3] Show models and training data
  [4] Show the rules
  [5] Quit
```

Type the number and press Enter. Pressing Enter alone takes the *(default)*.
Typing `q` at any prompt leaves the current screen.

## 3. Starting a game – step by step

### Step 1 – choose the table mode

```
Choose a game mode:
  [1] Standard – Sauspiel, Wenz and Solo; Ramsch when everybody passes (default)
  [2] Standard without Ramsch – cards are thrown in when everybody passes
  [3] Sauspiel only (beginner) – Ramsch when everybody passes
  [4] Ramsch only – nobody may announce, the most points lose
  [5] Custom – choose the allowed games yourself
```

The modes are explained in [game-modes.md](game-modes.md).

### Step 2 – number of rounds

Default 8 (every player deals twice). Any number from 1 to 1000.

### Step 3 – the players

For each of the four seats you pick a type and a name:

| Type            | Who plays                                                   |
|-----------------|-------------------------------------------------------------|
| Human           | you, at the keyboard                                        |
| AI – easy       | reinforcement-learning model `models/easy.npz`              |
| AI – medium     | reinforcement-learning model `models/medium.npz`            |
| AI – hard       | reinforcement-learning model `models/hard.npz`              |
| Rule-based bot  | hand-written heuristic strategy (no learning)              |
| Random bot      | random legal cards, rarely announces                       |

Any combination is allowed:

* **1 human + 3 AI** – the usual single-player game.
* **2–4 humans** – *hot-seat* mode: before a human's decision the program asks
  to hand the screen over and clears it, so nobody sees another player's hand.
* **0 humans** – watch the AI play. You are asked for a pause between moves
  (e.g. `0.5` seconds); `0` runs the match instantly.

If an AI tier has not been trained yet, the program says so and seats the
rule-based bot instead (see [ai-and-training.md](ai-and-training.md)).

## 4. A round on screen

### Bidding

```
═══ Round 1 ═══  dealer: Vroni

  Sepp, your hand:
    [1] ♣O Eichel-Ober
    [2] ♥O Herz-Ober
    [3] ♣U Eichel-Unter
    [4] ♥A Herz-Sau
    [5] ♥9 Herz-Neun
    [6] ♣K Eichel-König
    [7] ♠10 Gras-Zehn
    [8] ♦7 Schellen-Sieben
  What do you want to play?
    [1] Weiter (pass)
    [2] Sauspiel auf die Eichel-Sau
    [3] Sauspiel auf die Gras-Sau
    [4] Sauspiel auf die Schellen-Sau
    [5] Wenz
    [6] Eichel-Solo
    ...
  Your choice:
```

Only legal announcements are listed: you can only call a Sau whose suit you
hold and whose Sau you do not have, and you must outbid what was announced
before you. The hand is sorted: trumps first (strongest to weakest), then the
plain suits.

### Card play

```
  Game: Sauspiel auf die Gras-Sau (you play against the declarer)
  On the table: seat 2: ♠K Gras-König
  Sepp, your hand:
    [1] ♣O Eichel-Ober   (not allowed)
    [2] ♠A Gras-Sau
    [3] ♦7 Schellen-Sieben (not allowed)
  Card to play:
```

Cards you may not play (follow-suit and Rufsau rules) are marked
*(not allowed)*. Commands during card play:

| Input  | Effect                                           |
|--------|--------------------------------------------------|
| number | play that card                                   |
| `t`    | show tricks played so far and everybody's points |
| `h`    | help                                             |
| `q`    | stop the game (standings are shown)              |

Every card played and every trick taken is printed for everybody:

```
          Resi plays ♠K Gras-König
          Sepp plays ♠A Gras-Sau
          Wast plays ♠7 Gras-Sieben
         Vroni plays ♥U Herz-Unter
  → Vroni takes the trick (20 points)
```

### End of the round

```
  Sauspiel auf die Gras-Sau: Declarers win with 78 points – worth 10 ct per player
          Sepp:  31 points, 2 tricks, -0.10 €   total -0.10 €
          Resi:  47 points, 3 tricks, +0.10 €   total +0.10 €
          ...
```

After each round you press Enter for the next one. After the last round the
final standings and the winner are shown. Scoring is explained in
[winning-and-scoring.md](winning-and-scoring.md).

## 5. All commands

```bash
uv run schafskopf                       # interactive menu (same as `play`)
uv run schafskopf rules                 # print docs/rules.md
uv run schafskopf data                  # status of models and training vectors
uv run schafskopf train --tier medium   # train / continue training one tier
uv run schafskopf train-all             # train all three tiers
uv run schafskopf replay --tier hard    # re-learn from stored vectors
uv run schafskopf evaluate --tier hard  # benchmark a tier
```

Training and evaluation are described in [ai-and-training.md](ai-and-training.md).
Set `NO_COLOR=1` to disable colored cards.
