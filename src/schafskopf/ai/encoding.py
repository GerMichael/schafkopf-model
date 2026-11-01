"""Turn observations into fixed-length float vectors the networks can learn from.

All seat-dependent features are expressed *relative* to the observing seat
(offset 0 = me, 1 = next player clockwise, ...), so one network can play any
seat. See docs/ai-and-training.md for the full layout table.
"""

from __future__ import annotations

import numpy as np

from ..cards import ALL_CARDS, NUM_CARDS, Rank, Suit
from ..game import Observation
from ..players.heuristic import hand_strength
from ..rules import BID_ACTIONS, NUM_BIDS, PASS, Contract, GameType, effective_suit, is_trump, trick_winner

GAME_TYPES = (GameType.RAMSCH, GameType.SAUSPIEL, GameType.WENZ, GameType.SOLO)

# (name, size) blocks, in order. Kept as data so the docs and tests can use it.
PLAY_LAYOUT: tuple[tuple[str, int], ...] = (
    ("hand", 32),
    ("played_before", 32),
    ("trick_by_offset_1", 32),
    ("trick_by_offset_2", 32),
    ("trick_by_offset_3", 32),
    ("trump_cards", 32),
    ("game_type", 4),
    ("contract_suit", 4),
    ("declarer_offset", 4),
    ("relation_offsets_1_3", 6),
    ("rufsau_flags", 3),
    ("position_in_trick", 4),
    ("trick_number", 1),
    ("points_by_offset", 4),
    ("team_points", 2),
    ("trick_winner_offset", 4),
    ("trick_state", 2),
    ("trumps_unseen", 1),
)
BID_LAYOUT: tuple[tuple[str, int], ...] = (
    ("hand", 32),
    ("position_after_dealer", 4),
    ("highest_bid", 4),
    ("earlier_bids", 15),
    ("plain_suit_lengths", 4),
    ("ober_unter_ass", 3),
    ("solo_trump_counts", 4),
    ("sauspiel_trumps", 1),
    ("bid_strengths", NUM_BIDS),
)
PLAY_DIM = sum(n for _, n in PLAY_LAYOUT)
BID_DIM = sum(n for _, n in BID_LAYOUT)


def _offset(seat: int, other: int) -> int:
    return (other - seat) % 4


def encode_play(obs: Observation) -> np.ndarray:
    v = np.zeros(PLAY_DIM, dtype=np.float32)
    c = obs.contract
    seat = obs.seat
    i = 0

    for card in obs.hand:
        v[i + card.index] = 1.0
    i += 32
    for trick in obs.tricks_done:
        for _, card in trick:
            v[i + card.index] = 1.0
    i += 32
    for s, card in obs.trick:
        v[i + 32 * (_offset(seat, s) - 1) + card.index] = 1.0
    i += 96
    for card in ALL_CARDS:
        if is_trump(card, c):
            v[i + card.index] = 1.0
    i += 32

    v[i + GAME_TYPES.index(c.game_type)] = 1.0
    i += 4
    if c.suit is not None:
        v[i + int(c.suit)] = 1.0
    i += 4
    if c.declarer is not None:
        v[i + _offset(seat, c.declarer)] = 1.0
    i += 4
    for off in (1, 2, 3):
        rel = obs.relation((seat + off) % 4)
        if rel == 1:
            v[i] = 1.0
        elif rel == -1:
            v[i + 1] = 1.0
        i += 2
    v[i] = float(obs.own_partner is not None and seat != c.declarer)
    v[i + 1] = float(obs.called_suit_searched)
    v[i + 2] = float(obs.partner_revealed is not None)
    i += 3
    v[i + len(obs.trick)] = 1.0
    i += 4
    v[i] = obs.trick_number / 8.0
    i += 1
    for s in range(4):
        v[i + _offset(seat, s)] = obs.points[s] / 120.0
    i += 4
    mine = obs.points[seat] + sum(obs.points[s] for s in range(4) if obs.relation(s) == 1)
    theirs = sum(obs.points[s] for s in range(4) if obs.relation(s) == -1)
    v[i], v[i + 1] = mine / 120.0, theirs / 120.0
    i += 2
    if obs.trick:
        winner = trick_winner(obs.trick, c)
        v[i + _offset(seat, winner)] = 1.0
        v[i + 4] = sum(card.points for _, card in obs.trick) / 40.0
        v[i + 5] = float(obs.relation(winner))
    i += 6
    seen = set(obs.hand) | set(obs.played_cards)
    v[i] = sum(1 for card in ALL_CARDS if is_trump(card, c) and card not in seen) / 14.0
    i += 1
    assert i == PLAY_DIM
    return v


def encode_bid(obs: Observation) -> np.ndarray:
    v = np.zeros(BID_DIM, dtype=np.float32)
    hand = obs.hand
    i = 0
    for card in hand:
        v[i + card.index] = 1.0
    i += 32
    v[i + _offset(obs.dealer, obs.seat) - 1] = 1.0
    i += 4
    v[i + (0 if obs.highest is None else int(obs.highest))] = 1.0
    i += 4
    for s, b in obs.bids:
        bid = BID_ACTIONS[b]
        slot = 0 if b == PASS else int(bid.game_type)
        v[i + 5 * (_offset(obs.seat, s) - 1) + slot] = 1.0
    i += 15
    sauspiel = Contract(GameType.SAUSPIEL, 0, Suit.EICHEL)
    for card in hand:
        suit = effective_suit(card, sauspiel)
        if suit != "T":
            v[i + int(suit)] += 1 / 6
    i += 4
    v[i] = sum(c.rank is Rank.OBER for c in hand) / 4
    v[i + 1] = sum(c.rank is Rank.UNTER for c in hand) / 4
    v[i + 2] = sum(c.rank is Rank.ASS for c in hand) / 4
    i += 3
    for suit in Suit:
        solo = Contract(GameType.SOLO, 0, suit)
        v[i + int(suit)] = sum(is_trump(c, solo) for c in hand) / 8
    i += 4
    v[i] = sum(is_trump(c, sauspiel) for c in hand) / 8
    i += 1
    for b in range(1, NUM_BIDS):
        v[i + b] = hand_strength(hand, b)
    i += NUM_BIDS
    assert i == BID_DIM
    return v


def card_mask(legal) -> np.ndarray:
    mask = np.zeros(NUM_CARDS, dtype=bool)
    for card in legal:
        mask[card.index] = True
    return mask


def bid_mask(legal) -> np.ndarray:
    mask = np.zeros(NUM_BIDS, dtype=bool)
    mask[list(legal)] = True
    return mask
