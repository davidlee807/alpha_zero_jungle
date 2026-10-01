"""Train an AlphaZero agent for 香港鬥獸棋. Run from the alpha-zero-general folder:

    python main_jungle.py
"""
import logging

import coloredlogs

from Coach import Coach
from jungle.JungleGame import JungleGame as Game
from jungle.pytorch.NNet import NNetWrapper as nn
from utils import dotdict

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 200,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 10,       # Plies played with temperature 1 before switching to greedy moves.
    'updateThreshold': 0.55,   # The new net is kept if it wins at least this share of decided arena games.
    'maxlenOfQueue': 200000,   # Number of game examples to train the neural networks.
    'numMCTSSims': 100,         # Number of game moves for MCTS to simulate.
    'arenaCompare': 20,        # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.5,

    'checkpoint': './temp_jungle/',
    'load_model': False,
    'load_folder_file': ('./temp_jungle/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
