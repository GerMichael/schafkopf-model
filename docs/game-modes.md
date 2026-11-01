# Game Types and Table Modes

There are two different things you can "choose":

* the **game type** of a round (Sauspiel, Wenz, Solo, Ramsch) – decided by
  the players during bidding, every round anew;
* the **table mode** of a whole match – chosen once in the start menu, it
  decides which game types may be announced and what happens if everybody passes.

## Game types

### Sauspiel (Rufspiel) – 2 against 2

* Trumps: all Ober, all Unter and Herz (14 trumps).
* The declarer calls the Sau of Eichel, Gras or Schellen. The holder of that
  Sau is the partner; nobody (except the partner) knows who that is until the
  Sau is played.
* Declarer + partner need **61 points** together.
* Special duties of the Rufsau holder: see [rules.md §5](rules.md#5-the-called-ace-rufsau-in-a-sauspiel).
* Base value: 10 ct (paid by each loser to one winner).

### Wenz – 1 against 3

* Trumps: only the four Unter. Ober are normal suit cards (between König and Neun).
* The declarer plays alone and needs **61 points**.
* Base value: 50 ct, from/to each of the three opponents.
* Laufende are counted from 2 (only four trumps exist).

### Solo (Farbsolo) – 1 against 3

* Trumps: all Ober, all Unter and the chosen suit (Eichel, Gras, Herz or Schellen).
* The declarer plays alone and needs **61 points**.
* Base value: 50 ct, from/to each of the three opponents.

### Ramsch – everybody against everybody

* Only played when everybody passed and the table mode allows it.
* Trumps as in a Sauspiel (Ober, Unter, Herz).
* Nobody wants points: the player with the **most points loses**.
* Special cases: a player who takes all eight tricks wins a *Durchmarsch*;
  a player with no trick at all (*Jungfrau*) receives double. Details in
  [winning-and-scoring.md](winning-and-scoring.md#ramsch).

## Bidding priority

```
Pass  <  Sauspiel  <  Wenz  <  Solo
```

Each player announces once, in turn order, and must outbid the current
highest game type. The highest announcement is played. See [rules.md §6](rules.md#6-bidding-simplified).

## Table modes (start menu)

| # | Mode                       | Allowed announcements | Everybody passes        |
|---|----------------------------|-----------------------|-------------------------|
| 1 | Standard                   | Sauspiel, Wenz, Solo  | Ramsch                  |
| 2 | Standard without Ramsch    | Sauspiel, Wenz, Solo  | cards are thrown in     |
| 3 | Sauspiel only (beginner)   | Sauspiel              | Ramsch                  |
| 4 | Ramsch only                | none                  | Ramsch (every round)    |
| 5 | Custom                     | your choice           | your choice             |

"Thrown in" rounds count as a played round (the dealer moves on) but nobody pays.

The AI networks always see the same 9 possible announcements; game types the
mode forbids are simply masked out, so every trained model works in every mode.

## Programmatic use

```python
from schafskopf.game import GameSettings, Match
from schafskopf.rules import GameType
from schafskopf.players import HeuristicPlayer

settings = GameSettings(allowed=frozenset({GameType.SAUSPIEL, GameType.SOLO}),
                        ramsch_if_all_pass=False)
players = [HeuristicPlayer(n) for n in ("A", "B", "C", "D")]
print(Match(players, settings, rounds=100).play())   # totals in cents
```
