# Schafkopf – Rules

Schafkopf is the classic Bavarian trick-taking game for exactly **four players**.
This document describes the rules as implemented in this program. Where the
program simplifies traditional table rules, this is marked with **(simplified)**.

## 1. The deck

32 German-suited cards: four suits with eight ranks each.

| Suit      | Symbol on screen | Traditional ranking |
|-----------|------------------|---------------------|
| Eichel    | ♣                | 1 (highest)         |
| Gras      | ♠                | 2                   |
| Herz      | ♥                | 3                   |
| Schellen  | ♦                | 4 (lowest)          |

The suit ranking only matters among Ober and Unter (see trumps).

| Rank        | Code | Points ("Augen") |
|-------------|------|------------------|
| Sau (Ass)   | A    | 11               |
| Zehn        | 10   | 10               |
| König       | K    | 4                |
| Ober        | O    | 3                |
| Unter       | U    | 2                |
| Neun        | 9    | 0                |
| Acht        | 8    | 0                |
| Sieben      | 7    | 0                |

The whole deck is worth **120 points**. A card is shown as e.g. `♣O Eichel-Ober`.

## 2. Dealing and turn order

* Every player receives 8 cards.
* The dealer rotates clockwise every round (seat 1 → 2 → 3 → 4 → 1 …).
* The player to the left of the dealer (the next seat) bids first and leads
  the first trick. Play always proceeds clockwise.

## 3. Trumps

Which cards are trump depends on the game being played (see
[game-modes.md](game-modes.md)).

**Sauspiel, Herz-Solo and Ramsch** – 14 trumps, from highest to lowest:

```
Eichel-Ober, Gras-Ober, Herz-Ober, Schellen-Ober,
Eichel-Unter, Gras-Unter, Herz-Unter, Schellen-Unter,
Herz-Sau, Herz-Zehn, Herz-König, Herz-Neun, Herz-Acht, Herz-Sieben
```

**Farbsolo (Eichel / Gras / Schellen)** – the same, but the chosen suit
replaces Herz as trump suit. Herz becomes an ordinary suit.

**Wenz** – only the four Unter are trump:
`Eichel-Unter, Gras-Unter, Herz-Unter, Schellen-Unter`.
The Ober are ordinary suit cards and rank between König and Neun.

**Ordinary suits** (all non-trump cards) rank `Sau, Zehn, König, (Ober), Neun, Acht, Sieben`.
An Ober or Unter that is trump does **not** belong to its printed suit.
Example: in a Sauspiel the Eichel-Ober is a trump, not an Eichel card.

## 4. Playing a trick

1. The leader plays any card (subject to the Rufsau rules below).
2. Every other player must **follow**:
   * if a trump was led, play a trump if you have one;
   * if a suit was led, play a card of that suit if you have one.
3. If you cannot follow you may play **any** card – there is no obligation
   to trump or to beat the current card ("kein Stich-" or "Trumpfzwang").
4. The trick is won by the highest trump in it; if it contains no trump, by
   the highest card of the suit that was led. Cards of other suits never win.
5. The winner collects the cards (and their points) and leads the next trick.

After 8 tricks the round is scored ([winning-and-scoring.md](winning-and-scoring.md)).

## 5. The called ace (Rufsau) in a Sauspiel

In a Sauspiel the declarer calls the Sau (ace) of a suit. Whoever holds it is
the declarer's secret partner. Its holder has extra obligations until the called
suit has been played for the first time ("bis die Sau gesucht ist"):

* **Must answer the search:** when anyone leads the called suit, the holder
  must play the Rufsau.
* **May not discard it:** when a different suit or trump is led and the holder
  cannot follow, the Rufsau may not be thrown away – except on the very last
  trick, when it is the only card left.
* **Leading the called suit:** the holder may only lead the called suit with
  the Rufsau itself – unless they hold **four or more** cards of that suit.
  Then they may lead any of them and keep the Sau ("davonlaufen").

Once the called suit has been led once (or the Sau has fallen), these
restrictions end. The partnership becomes public the moment the Rufsau is played.

## 6. Bidding (simplified)

Starting left of the dealer, each player gets exactly one turn and either
passes ("weiter") or announces a game. An announcement must be of a **higher
game type** than everything announced before:

```
Sauspiel  <  Wenz  <  Solo
```

So after a Sauspiel only a Wenz or a Solo may follow, after a Wenz only a Solo,
and after a Solo everybody else must pass. The last (highest) announcement is
played. Because earlier players have priority at equal level in traditional
Schafkopf, a later player has to outbid with a higher type – this program
models exactly that, but without the traditional second negotiation round in
which an earlier player could upgrade their own game. **(simplified)**

Conditions for a Sauspiel announcement: you must hold at least one card of the
called suit (not counting Ober/Unter) and must **not** hold its Sau yourself.
Herz can never be called (it is trump).

If everybody passes, the table rule decides: either **Ramsch** is played, or
the cards are thrown in and the next dealer deals.

## 7. Not implemented

Kontra/Re, Klopfen/Stoß, Tout, Sie, Geier, Bettel and the short 24-card deck
are not part of this implementation. The tariff and Ramsch rules are
configurable, see [winning-and-scoring.md](winning-and-scoring.md).
