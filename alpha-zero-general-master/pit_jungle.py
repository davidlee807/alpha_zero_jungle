"""Play against, or test, a trained Jungle agent. Run from the alpha-zero-general folder:

    python pit_jungle.py                         # you vs the agent, in the terminal
    python pit_jungle.py --opponent greedy --games 20
    python pit_jungle.py --opponent random --games 20
"""
import argparse
import os

import numpy as np

import Arena
from MCTS import MCTS
from jungle.JungleGame import JungleGame
from jungle.JunglePlayers import GreedyJunglePlayer, HumanJunglePlayer, RandomPlayer
from jungle.pytorch.NNet import NNetWrapper as NNet
from utils import dotdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opponent', choices=['human', 'greedy', 'random'], default='human')
    ap.add_argument('--games', type=int, default=2, help='split evenly between both colours')
    ap.add_argument('--model', default='./temp_jungle/best.pth.tar')
    ap.add_argument('--sims', type=int, default=100)
    a = ap.parse_args()

    g = JungleGame()
    n1 = NNet(g)
    folder, filename = os.path.split(a.model)
    n1.load_checkpoint(folder or '.', filename)
    mcts1 = MCTS(g, n1, dotdict({'numMCTSSims': a.sims, 'cpuct': 1.5}))
    agent = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    opponent = {'human': HumanJunglePlayer, 'greedy': GreedyJunglePlayer,
                'random': RandomPlayer}[a.opponent](g).play
    verbose = a.opponent == 'human'
    arena = Arena.Arena(agent, opponent, g, display=JungleGame.display)
    agent_wins, opp_wins, draws = arena.playGames(a.games, verbose=verbose)
    print(f"agent wins {agent_wins}, {a.opponent} wins {opp_wins}, draws {draws}")


if __name__ == "__main__":
    main()
