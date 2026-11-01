"""Game types, trump order, follow-suit rules and trick evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from .cards import Card, Rank, Suit


class GameType(IntEnum):
    """Game types. The integer value is the bidding priority (higher wins)."""

    RAMSCH = 0
    SAUSPIEL = 1
    WENZ = 2
    SOLO = 3

    @property
    def label(self) -> str:
        return {
            GameType.RAMSCH: "Ramsch",
            GameType.SAUSPIEL: "Sauspiel",
            GameType.WENZ: "Wenz",
            GameType.SOLO: "Solo",
        }[self]


@dataclass(frozen=True, slots=True)
class Bid:
    """A possible announcement during bidding. ``None`` game type means pass."""

    game_type: GameType | None
    suit: Suit | None = None

    @property
    def is_pass(self) -> bool:
        return self.game_type is None

    @property
    def label(self) -> str:
        if self.game_type is None:
            return "Weiter (pass)"
        if self.game_type is GameType.SAUSPIEL:
            return f"Sauspiel auf die {self.suit.label}-Sau"
        if self.game_type is GameType.SOLO:
            return f"{self.suit.label}-Solo"
        return self.game_type.label


# The fixed action space of the bidding network. Index 0 is always "pass".
BID_ACTIONS: tuple[Bid, ...] = (
    Bid(None),
    Bid(GameType.SAUSPIEL, Suit.EICHEL),
    Bid(GameType.SAUSPIEL, Suit.GRAS),
    Bid(GameType.SAUSPIEL, Suit.SCHELLEN),
    Bid(GameType.WENZ),
    Bid(GameType.SOLO, Suit.EICHEL),
    Bid(GameType.SOLO, Suit.GRAS),
    Bid(GameType.SOLO, Suit.HERZ),
    Bid(GameType.SOLO, Suit.SCHELLEN),
)
NUM_BIDS = len(BID_ACTIONS)
PASS = 0


@dataclass(frozen=True, slots=True)
class Contract:
    """The game being played in a round.

    ``suit`` is the called suit for a Sauspiel and the trump suit for a Solo.
    ``declarer`` is ``None`` for Ramsch.
    """

    game_type: GameType
    declarer: int | None = None
    suit: Suit | None = None

    @property
    def trump_suit(self) -> Suit | None:
        if self.game_type in (GameType.SAUSPIEL, GameType.RAMSCH):
            return Suit.HERZ
        if self.game_type is GameType.SOLO:
            return self.suit
        return None  # Wenz: only the Unter are trump

    @property
    def called_ace(self) -> Card | None:
        if self.game_type is GameType.SAUSPIEL and self.suit is not None:
            return Card(self.suit, Rank.ASS)
        return None

    @property
    def label(self) -> str:
        if self.game_type is GameType.SAUSPIEL:
            return f"Sauspiel auf die {self.suit.label}-Sau"
        if self.game_type is GameType.SOLO:
            return f"{self.suit.label}-Solo"
        return self.game_type.label

    @classmethod
    def from_bid(cls, bid: Bid, declarer: int) -> Contract:
        return cls(bid.game_type, declarer, bid.suit)


# Order of the plain (non-trump) ranks, strongest first.
PLAIN_ORDER = (Rank.ASS, Rank.ZEHN, Rank.KOENIG, Rank.OBER, Rank.UNTER, Rank.NEUN, Rank.ACHT, Rank.SIEBEN)
SUIT_ORDER = (Suit.EICHEL, Suit.GRAS, Suit.HERZ, Suit.SCHELLEN)
TRUMP = "T"  # pseudo-suit returned by :func:`effective_suit` for trumps


def _trump_ranks(game_type: GameType) -> tuple[Rank, ...]:
    """Ranks that are trump regardless of suit (Ober/Unter or only Unter)."""
    if game_type is GameType.WENZ:
        return (Rank.UNTER,)
    return (Rank.OBER, Rank.UNTER)


def is_trump(card: Card, contract: Contract) -> bool:
    if card.rank in _trump_ranks(contract.game_type):
        return True
    return card.suit == contract.trump_suit


def effective_suit(card: Card, contract: Contract) -> Suit | str:
    """The suit a card counts as when following: its suit, or ``TRUMP``."""
    return TRUMP if is_trump(card, contract) else card.suit


def trump_order(contract: Contract) -> list[Card]:
    """All trumps of a contract, strongest first."""
    order: list[Card] = []
    for rank in _trump_ranks(contract.game_type):
        order.extend(Card(s, rank) for s in SUIT_ORDER)
    if contract.trump_suit is not None:
        special = _trump_ranks(contract.game_type)
        order.extend(Card(contract.trump_suit, r) for r in PLAIN_ORDER if r not in special)
    return order


def card_power(card: Card, contract: Contract, led: Suit | str) -> int:
    """Strength of a card in a trick; higher wins. ``-1`` cannot win."""
    if is_trump(card, contract):
        order = trump_order(contract)
        return 100 + (len(order) - order.index(card))
    if card.suit == led:
        plain = [r for r in PLAIN_ORDER if r not in _trump_ranks(contract.game_type)]
        return len(plain) - plain.index(card.rank)
    return -1


def trick_winner(trick: Sequence[tuple[int, Card]], contract: Contract) -> int:
    """Seat that wins the trick. ``trick`` is a list of ``(seat, card)``."""
    led = effective_suit(trick[0][1], contract)
    best_seat, best_power = trick[0][0], -2
    for seat, card in trick:
        power = card_power(card, contract, led)
        if power > best_power:
            best_seat, best_power = seat, power
    return best_seat


def legal_cards(
    hand: Sequence[Card],
    trick: Sequence[tuple[int, Card]],
    contract: Contract,
    called_suit_searched: bool = False,
) -> list[Card]:
    """Cards the owner of ``hand`` may play into the current ``trick``.

    Implements following suit/trump and the special obligations of the
    holder of the called ace (Rufsau) in a Sauspiel:

    * when the called suit is led, the Rufsau must be played;
    * the Rufsau must not be discarded on another suit (except on the last trick);
    * when leading the called suit, the holder must lead the Rufsau unless
      they hold four or more cards of that suit ("davonlaufen").
    """
    hand = list(hand)
    ace = contract.called_ace
    holds_ace = ace is not None and ace in hand and not called_suit_searched

    if not trick:
        if not holds_ace:
            return hand
        called = [c for c in hand if effective_suit(c, contract) == ace.suit]
        if len(called) >= 4:
            return hand  # may run away with any card of the called suit
        return [c for c in hand if c == ace or c not in called]

    led = effective_suit(trick[0][1], contract)
    following = [c for c in hand if effective_suit(c, contract) == led]
    if following:
        if holds_ace and led == ace.suit:
            return [ace]
        return following
    if holds_ace and len(hand) > 1:
        return [c for c in hand if c != ace]
    return hand


def legal_bids(hand: Sequence[Card], highest: GameType | None, allowed: set[GameType]) -> list[int]:
    """Indices into :data:`BID_ACTIONS` that are legal for this hand.

    A player may only announce a game type that ranks strictly above the
    highest announcement so far. A Sauspiel requires holding at least one
    non-trump card of the called suit and not holding its ace.
    """
    result = [PASS]
    for i, bid in enumerate(BID_ACTIONS):
        if bid.is_pass or bid.game_type not in allowed:
            continue
        if highest is not None and bid.game_type <= highest:
            continue
        if bid.game_type is GameType.SAUSPIEL:
            contract = Contract(GameType.SAUSPIEL, 0, bid.suit)
            if Card(bid.suit, Rank.ASS) in hand:
                continue
            if not any(effective_suit(c, contract) == bid.suit for c in hand):
                continue
        result.append(i)
    return result
