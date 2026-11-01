# Winning Conditions and Scoring

Everything is counted in **cents**. Each round is zero-sum: what the losers pay,
the winners receive. The match totals are shown in euros.

## 1. Who wins a round?

| Game type        | Declaring side             | Wins with           |
|------------------|----------------------------|---------------------|
| Sauspiel         | declarer + Rufsau holder   | **61** or more points together |
| Wenz, Solo       | declarer alone             | **61** or more points          |
| Ramsch           | –                          | the player with the most points **loses** |

With exactly 60 points the declaring side **loses** (the opponents only need 60).

## 2. Value of a won or lost game

```
value = base + Schneider + Schwarz + Laufende × laufende tariff
```

| Component   | Condition                                                         | Default |
|-------------|-------------------------------------------------------------------|---------|
| base        | Sauspiel / Wenz or Solo                                           | 10 / 50 ct |
| Schneider   | the losing side has **30 points or fewer**                         | +10 ct  |
| Schwarz     | the losing side took **no trick at all**                           | +10 ct  |
| Laufende    | number of top trumps held in one unbroken run by the same side, from the highest trump down; counted from 3 (Sauspiel, Solo) or 2 (Wenz) | +10 ct each |

Notes

* Schneider and Schwarz apply in both directions – if the declarers lose with
  30 points or fewer, they lose "Schneider".
* Laufende count for whichever side holds them ("mit" or "ohne"). In a
  Sauspiel/Solo the run starts at Eichel-Ober; with Eichel-Ober, Gras-Ober and
  Herz-Ober on one side the game has "drei Laufende". Laufende are determined
  from the hands as dealt.

### Example

Sauspiel, the declarers hold the three highest Ober and win with 95 points
(opponents 25 → Schneider): `10 + 10 + 3 × 10 = 50 ct`.

## 3. Who pays whom?

* **Sauspiel**: each player of the losing pair pays the value to one winner.
  Payouts: winners `+value` each, losers `−value` each.
* **Wenz / Solo**: each of the three opponents pays (or receives) the value.
  Payouts: declarer `±3 × value`, each opponent `∓value`.

## 4. Ramsch

* All points count for each player individually.
* The player with the **most points** loses and pays the Ramsch tariff
  (10 ct) to each other player. On a tie, all tied players lose and each pays
  every non-loser.
* **Jungfrau**: a player who took no trick receives double from each loser.
* **Durchmarsch**: a player who takes all eight tricks wins and receives
  `3 × Ramsch tariff` from each other player.

## 5. Thrown-in rounds

If everybody passes and Ramsch is disabled, nobody pays and the next dealer deals.

## 6. Winning the match

A match lasts the number of rounds chosen in the start menu (default 8 – two
full turns of dealing). After the last round the players are ranked by their
money balance; the player(s) with the highest balance win. You can stop a
match at any time with `q`; the standings at that moment are shown.

## 7. Changing the tariff

The tariff is the `Tariff` dataclass in
[`src/schafskopf/scoring.py`](../src/schafskopf/scoring.py):

```python
Tariff(sauspiel=10, solo=50, ramsch=10, schneider=10, schwarz=10,
       laufende=10, min_laufende=3, min_laufende_wenz=2)
```

Pass your own via `GameSettings(tariff=Tariff(sauspiel=20, solo=100))` or edit
the defaults. Note that the AI's reward is derived from the payouts, so a
different tariff changes what the AI considers valuable (see `reward_scale`
in [ai-and-training.md](ai-and-training.md)).
