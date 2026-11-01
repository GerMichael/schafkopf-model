import random

from schafskopf.cards import ALL_CARDS, Card, Rank, Suit
from schafskopf.game import GameSettings, Round
from schafskopf.players import HeuristicPlayer, RandomPlayer
from schafskopf.rules import (
    BID_ACTIONS,
    PASS,
    Contract,
    GameType,
    legal_bids,
    legal_cards,
    trick_winner,
    trump_order,
)
from schafskopf.scoring import Tariff, count_laufende, score_round

E, G, H, S = Suit.EICHEL, Suit.GRAS, Suit.HERZ, Suit.SCHELLEN
A, Z, K, O, U, N9, N8, N7 = Rank


def c(suit, rank):
    return Card(suit, rank)


SAUSPIEL_GRAS = Contract(GameType.SAUSPIEL, 0, G)


def test_deck_has_120_points():
    assert len(ALL_CARDS) == 32
    assert sum(card.points for card in ALL_CARDS) == 120


def test_trump_orders():
    order = trump_order(SAUSPIEL_GRAS)
    assert len(order) == 14
    assert order[:4] == [c(E, O), c(G, O), c(H, O), c(S, O)]
    assert order[4:8] == [c(E, U), c(G, U), c(H, U), c(S, U)]
    assert order[8:] == [c(H, A), c(H, Z), c(H, K), c(H, N9), c(H, N8), c(H, N7)]
    assert trump_order(Contract(GameType.WENZ, 0)) == [c(E, U), c(G, U), c(H, U), c(S, U)]
    assert len(trump_order(Contract(GameType.SOLO, 0, S))) == 14


def test_trick_winner_trump_beats_ace():
    trick = [(0, c(E, A)), (1, c(H, N7)), (2, c(E, Z)), (3, c(E, K))]
    assert trick_winner(trick, SAUSPIEL_GRAS) == 1


def test_trick_winner_off_suit_never_wins():
    trick = [(0, c(E, N7)), (1, c(G, A)), (2, c(S, A)), (3, c(E, N8))]
    assert trick_winner(trick, SAUSPIEL_GRAS) == 3


def test_wenz_ober_is_plain_card():
    wenz = Contract(GameType.WENZ, 0)
    trick = [(0, c(E, K)), (1, c(E, O)), (2, c(H, A)), (3, c(S, U))]
    assert trick_winner(trick, wenz) == 3
    trick = [(0, c(E, N9)), (1, c(E, O)), (2, c(H, A)), (3, c(E, K))]
    assert trick_winner(trick, wenz) == 3  # König beats Ober
    trick = [(0, c(E, N9)), (1, c(E, O)), (2, c(H, A)), (3, c(E, N7))]
    assert trick_winner(trick, wenz) == 1  # Ober ranks between König and Neun


def test_must_follow_trump_with_ober():
    hand = [c(E, O), c(E, A), c(G, N7)]
    trick = [(1, c(H, N9))]
    assert legal_cards(hand, trick, SAUSPIEL_GRAS) == [c(E, O)]


def test_ober_does_not_follow_suit():
    hand = [c(E, O), c(G, N7)]
    trick = [(1, c(E, A))]
    assert set(legal_cards(hand, trick, SAUSPIEL_GRAS)) == set(hand)


def test_rufsau_must_be_played_when_searched():
    hand = [c(G, A), c(G, N7), c(E, N8)]
    trick = [(1, c(G, K))]
    assert legal_cards(hand, trick, SAUSPIEL_GRAS) == [c(G, A)]


def test_rufsau_may_not_be_discarded():
    hand = [c(G, A), c(G, N7), c(E, N8)]
    trick = [(1, c(S, K))]
    assert c(G, A) not in legal_cards(hand, trick, SAUSPIEL_GRAS)


def test_rufsau_lead_and_davonlaufen():
    hand = [c(G, A), c(G, N7), c(E, N8)]
    legal = legal_cards(hand, [], SAUSPIEL_GRAS)
    assert c(G, N7) not in legal and c(G, A) in legal
    long_hand = [c(G, A), c(G, N7), c(G, N8), c(G, N9), c(E, N8)]
    assert set(legal_cards(long_hand, [], SAUSPIEL_GRAS)) == set(long_hand)


def test_rufsau_free_after_suit_searched():
    hand = [c(G, A), c(G, N7), c(E, N8)]
    trick = [(1, c(S, K))]
    assert c(G, A) in legal_cards(hand, trick, SAUSPIEL_GRAS, called_suit_searched=True)


def test_sauspiel_bid_needs_called_suit_without_ace():
    hand = [c(G, A), c(E, N7), c(H, O)]
    allowed = {GameType.SAUSPIEL, GameType.WENZ, GameType.SOLO}
    legal = [BID_ACTIONS[i] for i in legal_bids(hand, None, allowed)]
    called = {b.suit for b in legal if b.game_type is GameType.SAUSPIEL}
    assert called == {E}


def test_bids_must_outrank_highest():
    hand = [c(E, N7), c(G, N7)]
    allowed = {GameType.SAUSPIEL, GameType.WENZ, GameType.SOLO}
    legal = [BID_ACTIONS[i] for i in legal_bids(hand, GameType.WENZ, allowed)]
    assert all(b.is_pass or b.game_type is GameType.SOLO for b in legal)


def test_laufende_counts_top_trumps():
    hands = [[c(E, O), c(G, O), c(H, O)], [c(S, O)], [], []]
    rest = [card for card in ALL_CARDS if card not in hands[0] + hands[1]]
    hands[2], hands[3] = rest[:14], rest[14:]
    assert count_laufende(SAUSPIEL_GRAS, hands, {0, 2}) >= 3
    # "ohne": the opponents hold the top trumps – also counts
    assert count_laufende(SAUSPIEL_GRAS, hands, {1}) >= 3


def test_scoring_sauspiel_win_and_schneider():
    hands = [list(ALL_CARDS[i * 8:(i + 1) * 8]) for i in range(4)]
    result = score_round(SAUSPIEL_GRAS, hands, [70, 20, 25, 5], [4, 1, 2, 1], partner=2,
                         tariff=Tariff(laufende=0))
    assert result.declarers_won and result.schneider
    assert result.payouts == [20, -20, 20, -20]


def test_scoring_solo_pays_triple():
    solo = Contract(GameType.SOLO, 1, H)
    hands = [list(ALL_CARDS[i * 8:(i + 1) * 8]) for i in range(4)]
    result = score_round(solo, hands, [20, 60, 20, 20], [2, 3, 2, 1], partner=None, tariff=Tariff(laufende=0))
    assert not result.declarers_won  # 60 is not enough
    assert result.payouts == [50, -150, 50, 50]


def test_ramsch_loser_pays():
    ramsch = Contract(GameType.RAMSCH)
    result = score_round(ramsch, [[]] * 4, [50, 30, 25, 15], [3, 2, 2, 1], partner=None)
    assert result.ramsch_losers == (0,)
    assert result.payouts == [-30, 10, 10, 10]


def test_full_rounds_are_consistent():
    rng = random.Random(7)
    players = [HeuristicPlayer("a", rng), RandomPlayer("b", rng), HeuristicPlayer("c", rng), RandomPlayer("d", rng)]
    for i in range(300):
        result = Round(players, i % 4, GameSettings(), rng).run()
        if result is not None:
            assert sum(result.payouts) == 0
            assert sum(result.points) == 120
            assert sum(result.tricks) == 8


def test_bid_index_zero_is_pass():
    assert BID_ACTIONS[PASS].is_pass
