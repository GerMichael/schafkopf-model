"""The game engine: dealing, bidding, trick play, scoring and matches."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from .cards import Card, deal
from .rules import (
    BID_ACTIONS,
    PASS,
    Bid,
    Contract,
    GameType,
    effective_suit,
    legal_bids,
    legal_cards,
    trick_winner,
)
from .scoring import RoundResult, Tariff, score_round

if TYPE_CHECKING:
    from .players.base import Player


@dataclass(frozen=True, slots=True)
class GameSettings:
    """Table rules chosen in the start menu."""

    allowed: frozenset[GameType] = frozenset({GameType.SAUSPIEL, GameType.WENZ, GameType.SOLO})
    ramsch_if_all_pass: bool = True
    tariff: Tariff = Tariff()


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything one seat is allowed to know at the moment of a decision."""

    seat: int
    dealer: int
    hand: tuple[Card, ...]
    bids: tuple[tuple[int, int], ...]  # (seat, index into BID_ACTIONS)
    highest: GameType | None
    contract: Contract | None
    trick: tuple[tuple[int, Card], ...]
    tricks_done: tuple[tuple[tuple[int, Card], ...], ...]
    points: tuple[int, int, int, int]
    tricks_won: tuple[int, int, int, int]
    called_suit_searched: bool
    partner_revealed: int | None  # public: seat that played the called ace
    own_partner: int | None  # private knowledge of this seat's partner

    @property
    def trick_number(self) -> int:
        return len(self.tricks_done)

    @property
    def played_cards(self) -> list[Card]:
        cards = [card for trick in self.tricks_done for _, card in trick]
        cards.extend(card for _, card in self.trick)
        return cards

    def relation(self, other: int) -> int:
        """+1 known teammate, -1 known opponent, 0 unknown (or self)."""
        if other == self.seat or self.contract is None:
            return 0
        c = self.contract
        if c.game_type is GameType.RAMSCH:
            return -1
        if c.game_type is not GameType.SAUSPIEL:
            if self.seat == c.declarer:
                return -1
            return -1 if other == c.declarer else 1
        partner = self.own_partner if self.own_partner is not None else self.partner_revealed
        declarer_side = {c.declarer} | ({partner} if partner is not None else set())
        if partner is None:
            # Partner still hidden: only the declarer is surely on the other side.
            if self.seat == c.declarer:
                return 0
            return -1 if other == c.declarer else 0
        mine = self.seat in declarer_side
        return 1 if (other in declarer_side) == mine else -1


class Observer(Protocol):
    """Receives public game events, e.g. to render them on the console."""

    def round_started(self, number: int, dealer: int, players: Sequence[Player]) -> None: ...
    def bid_made(self, seat: int, bid: Bid) -> None: ...
    def contract_set(self, contract: Contract | None) -> None: ...
    def card_played(self, seat: int, card: Card) -> None: ...
    def trick_completed(self, trick: Sequence[tuple[int, Card]], winner: int, points: int) -> None: ...
    def round_finished(self, result: RoundResult | None, totals: Sequence[int]) -> None: ...


class Round:
    """One deal: bidding, eight tricks, scoring."""

    def __init__(
        self,
        players: Sequence[Player],
        dealer: int,
        settings: GameSettings = GameSettings(),
        rng: random.Random | None = None,
        observers: Sequence[Observer] = (),
        hands: Sequence[Sequence[Card]] | None = None,
    ) -> None:
        self.players = list(players)
        self.dealer = dealer
        self.settings = settings
        self.observers = list(observers)
        self.hands = [list(h) for h in hands] if hands is not None else deal(rng)
        self.initial_hands = [list(h) for h in self.hands]
        self.bids: list[tuple[int, int]] = []
        self.highest: GameType | None = None
        self.contract: Contract | None = None
        self.partner: int | None = None
        self.partner_revealed: int | None = None
        self.called_suit_searched = False
        self.trick: list[tuple[int, Card]] = []
        self.tricks_done: list[tuple[tuple[int, Card], ...]] = []
        self.points = [0, 0, 0, 0]
        self.tricks_won = [0, 0, 0, 0]

    # -- observations -----------------------------------------------------
    def observe(self, seat: int) -> Observation:
        own_partner = None
        if self.contract is not None and self.contract.game_type is GameType.SAUSPIEL:
            if seat == self.partner:
                own_partner = self.contract.declarer
            elif seat == self.contract.declarer and self.partner_revealed is not None:
                own_partner = self.partner
        return Observation(
            seat=seat,
            dealer=self.dealer,
            hand=tuple(self.hands[seat]),
            bids=tuple(self.bids),
            highest=self.highest,
            contract=self.contract,
            trick=tuple(self.trick),
            tricks_done=tuple(self.tricks_done),
            points=tuple(self.points),
            tricks_won=tuple(self.tricks_won),
            called_suit_searched=self.called_suit_searched,
            partner_revealed=self.partner_revealed,
            own_partner=own_partner,
        )

    def _emit(self, method: str, *args) -> None:
        for obs in self.observers:
            getattr(obs, method)(*args)

    # -- phases -----------------------------------------------------------
    def run_bidding(self) -> Contract | None:
        declarer, best = None, None
        for offset in range(1, 5):
            seat = (self.dealer + offset) % 4
            legal = legal_bids(self.hands[seat], self.highest, set(self.settings.allowed))
            choice = self.players[seat].bid(self.observe(seat), legal)
            if choice not in legal:
                raise ValueError(f"{self.players[seat].name} made an illegal bid {choice}")
            self.bids.append((seat, choice))
            bid = BID_ACTIONS[choice]
            self._emit("bid_made", seat, bid)
            if choice != PASS:
                declarer, best = seat, bid
                self.highest = bid.game_type

        if best is not None:
            self.contract = Contract.from_bid(best, declarer)
            if best.game_type is GameType.SAUSPIEL:
                ace = self.contract.called_ace
                self.partner = next(s for s in range(4) if ace in self.hands[s])
        elif self.settings.ramsch_if_all_pass:
            self.contract = Contract(GameType.RAMSCH)
        self._emit("contract_set", self.contract)
        return self.contract

    def run_play(self) -> None:
        contract = self.contract
        leader = (self.dealer + 1) % 4
        for _ in range(8):
            self.trick = []
            for offset in range(4):
                seat = (leader + offset) % 4
                legal = legal_cards(self.hands[seat], self.trick, contract, self.called_suit_searched)
                card = self.players[seat].play(self.observe(seat), legal)
                if card not in legal:
                    raise ValueError(f"{self.players[seat].name} played an illegal card {card}")
                self.hands[seat].remove(card)
                if not self.trick and contract.called_ace is not None:
                    if effective_suit(card, contract) == contract.called_ace.suit:
                        self.called_suit_searched = True
                if card == contract.called_ace:
                    self.partner_revealed = seat
                    self.called_suit_searched = True
                self.trick.append((seat, card))
                self._emit("card_played", seat, card)
            winner = trick_winner(self.trick, contract)
            gained = sum(card.points for _, card in self.trick)
            self.points[winner] += gained
            self.tricks_won[winner] += 1
            self.tricks_done.append(tuple(self.trick))
            self._emit("trick_completed", list(self.trick), winner, gained)
            leader = winner
        self.trick = []

    def run(self) -> RoundResult | None:
        """Play the whole round. Returns ``None`` if the cards were thrown in."""
        if self.run_bidding() is None:
            result = None
        else:
            self.run_play()
            result = score_round(
                self.contract, self.initial_hands, self.points, self.tricks_won,
                self.partner, self.settings.tariff,
            )
        for seat, player in enumerate(self.players):
            player.round_finished(self.observe(seat), result)
        return result


@dataclass
class Match:
    """A series of rounds with rotating dealer and running totals (in cents)."""

    players: Sequence[Player]
    settings: GameSettings = field(default_factory=GameSettings)
    rounds: int = 8
    rng: random.Random = field(default_factory=random.Random)
    observers: Sequence[Observer] = ()
    totals: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    history: list[RoundResult | None] = field(default_factory=list)
    dealer: int = 0

    def play_round(self) -> RoundResult | None:
        for obs in self.observers:
            obs.round_started(len(self.history) + 1, self.dealer, self.players)
        result = Round(self.players, self.dealer, self.settings, self.rng, self.observers).run()
        if result is not None:
            for seat, amount in enumerate(result.payouts):
                self.totals[seat] += amount
        self.history.append(result)
        for obs in self.observers:
            obs.round_finished(result, self.totals)
        self.dealer = (self.dealer + 1) % 4
        return result

    def play(self) -> list[int]:
        for _ in range(self.rounds):
            self.play_round()
        return self.totals

    def winners(self) -> list[int]:
        best = max(self.totals)
        return [s for s in range(4) if self.totals[s] == best]
