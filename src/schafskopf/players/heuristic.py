"""A rule-based player. Used as baseline opponent for training and evaluation."""

from __future__ import annotations

import random

from ..cards import Card, Rank, Suit
from ..game import Observation
from ..rules import (
    BID_ACTIONS,
    PASS,
    Contract,
    GameType,
    card_power,
    effective_suit,
    is_trump,
    trick_winner,
)
from .base import Player


def hand_strength(hand: tuple[Card, ...] | list[Card], bid_index: int) -> float:
    """Rough hand evaluation for a bid; >= 1.0 means "worth playing"."""
    bid = BID_ACTIONS[bid_index]
    contract = Contract(bid.game_type, 0, bid.suit)
    trumps = [c for c in hand if is_trump(c, contract)]
    obers = sum(c.rank is Rank.OBER for c in hand)
    unters = sum(c.rank is Rank.UNTER for c in hand)
    side_aces = sum(c.rank is Rank.ASS and not is_trump(c, contract) for c in hand)
    high = sum(c in (Card(Suit.EICHEL, Rank.OBER), Card(Suit.GRAS, Rank.OBER)) for c in hand)

    if bid.game_type is GameType.SAUSPIEL:
        called = [c for c in hand if effective_suit(c, contract) == bid.suit]
        score = len(trumps) / 4.5 + (obers + unters) * 0.08 + side_aces * 0.1
        score += 0.1 if len(called) == 1 else 0.0
        return score
    if bid.game_type is GameType.SOLO:
        return len(trumps) / 6.5 + high * 0.12 + obers * 0.05 + side_aces * 0.08 - 0.15
    if bid.game_type is GameType.WENZ:
        eu = Card(Suit.EICHEL, Rank.UNTER) in hand
        return unters * 0.22 + side_aces * 0.15 + (0.12 if eu else -0.1)
    return 0.0


class HeuristicPlayer(Player):
    def __init__(self, name: str, rng: random.Random | None = None, threshold: float = 1.0) -> None:
        super().__init__(name)
        self.rng = rng or random.Random()
        self.threshold = threshold

    @property
    def kind(self) -> str:
        return "Heuristic"

    # -- bidding ----------------------------------------------------------
    def bid(self, obs: Observation, legal: list[int]) -> int:
        scored = [(hand_strength(obs.hand, b), BID_ACTIONS[b].game_type, b) for b in legal if b != PASS]
        playable = [s for s in scored if s[0] >= self.threshold]
        if not playable:
            return PASS
        # prefer the highest game type the hand supports, then the best fit
        playable.sort(key=lambda s: (s[1], s[0]), reverse=True)
        return playable[0][2]

    # -- card play --------------------------------------------------------
    def play(self, obs: Observation, legal: list[Card]) -> Card:
        if len(legal) == 1:
            return legal[0]
        contract = obs.contract
        if contract.game_type is GameType.RAMSCH:
            return self._play_ramsch(obs, legal)
        if not obs.trick:
            return self._lead(obs, legal)
        return self._follow(obs, legal)

    def _on_declarer_side(self, obs: Observation) -> bool:
        c = obs.contract
        return obs.seat == c.declarer or obs.own_partner == c.declarer

    def _lead(self, obs: Observation, legal: list[Card]) -> Card:
        c = obs.contract
        trumps = sorted((x for x in legal if is_trump(x, c)), key=lambda x: card_power(x, c, "T"), reverse=True)
        plain = [x for x in legal if not is_trump(x, c)]
        aces = [x for x in plain if x.rank is Rank.ASS]

        if self._on_declarer_side(obs):
            if trumps and len(trumps) >= 3:
                return trumps[0]
            if aces:
                return aces[0]
            if trumps:
                return trumps[-1]
            return min(plain, key=lambda x: x.points)

        # opponents: search the called ace, otherwise cash aces, otherwise go low
        if c.called_ace is not None and not obs.called_suit_searched:
            search = [x for x in plain if x.suit == c.called_ace.suit]
            if search:
                return min(search, key=lambda x: x.points)
        if aces:
            return aces[0]
        if plain:
            return min(plain, key=lambda x: x.points)
        return trumps[-1]

    def _follow(self, obs: Observation, legal: list[Card]) -> Card:
        c = obs.contract
        trick = list(obs.trick)
        led = effective_suit(trick[0][1], c)
        current = trick_winner(trick, c)
        current_power = max(card_power(card, c, led) for _, card in trick)
        last_to_play = len(trick) == 3
        mate_winning = obs.relation(current) == 1
        on_table = sum(card.points for _, card in trick)

        def cost(card: Card) -> float:
            return card.points + (12 if is_trump(card, c) else 0) + card_power(card, c, led) / 200

        if mate_winning and (last_to_play or current_power > 108):
            # partner holds the trick: give points ("schmieren")
            return max(legal, key=lambda x: x.points - (12 if is_trump(x, c) else 0))

        winners = [x for x in legal if card_power(x, c, led) > current_power]
        if winners and (on_table >= 10 or last_to_play or led != "T"):
            return min(winners, key=lambda x: card_power(x, c, led) + (0 if last_to_play else 0))
        return min(legal, key=cost)

    def _play_ramsch(self, obs: Observation, legal: list[Card]) -> Card:
        c = obs.contract
        if not obs.trick:
            return min(legal, key=lambda x: card_power(x, c, effective_suit(x, c)) + x.points)
        led = effective_suit(obs.trick[0][1], c)
        current_power = max(card_power(card, c, led) for _, card in obs.trick)
        losing = [x for x in legal if card_power(x, c, led) < current_power]
        if losing:
            return max(losing, key=lambda x: x.points)
        return max(legal, key=lambda x: card_power(x, c, led) - x.points)
