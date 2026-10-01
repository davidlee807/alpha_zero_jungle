"""Lets jungle_gui.py play against a trained agent (python jungle_gui.py --ai top)."""
import os

import numpy as np

import jungle_engine as E
from MCTS import MCTS
from jungle.JungleGame import JungleGame
from jungle.JungleLogic import DIRS
from jungle.pytorch.NNet import NNetWrapper
from utils import dotdict


class AlphaZeroBot:
    def __init__(self, model_path, sims=100, cpuct=1.5):
        self.game = JungleGame()
        self.net = NNetWrapper(self.game)
        folder, filename = os.path.split(model_path)
        self.net.load_checkpoint(folder or '.', filename)
        self.args = dotdict({'numMCTSSims': sims, 'cpuct': cpuct})
        self.reset()

    def reset(self):
        self.mcts = MCTS(self.game, self.net, self.args)

    def to_board(self, state, log):
        b = np.zeros((E.ROWS + 1, E.COLS), dtype=np.int16)
        for i, p in enumerate(state.board):
            if p is not None:
                b[i // E.COLS, i % E.COLS] = p.rank if p.side == E.SOUTH else -p.rank
        since = 0
        for m in reversed(log):
            if m.capture:
                break
            since += 1
        # The GUI has no draw limit; keep the counters short of the training limits
        # so the agent keeps playing for a win.
        b[E.ROWS, 0] = min(since, self.game.no_capture_limit - 10)
        b[E.ROWS, 1] = min(len(log), self.game.max_plies - 10)
        return b

    def choose(self, state, log, legal):
        """Pick an engine Move for the side to move in `state`."""
        player = 1 if state.turn == E.SOUTH else -1
        canonical = self.game.getCanonicalForm(self.to_board(state, log), player)
        # temp=1 returns visit shares; walk them from most visited down, so a move
        # the GUI's current house rules forbid is skipped.
        probs = np.asarray(self.mcts.getActionProb(canonical, temp=1))
        for a in np.argsort(-probs, kind='stable'):
            move = self._to_engine(int(a), player, legal)
            if move is not None:
                return move
        return legal[0]

    @staticmethod
    def _to_engine(action, player, legal):
        sq, d = divmod(action, 4)
        r, c = divmod(sq, E.COLS)
        dr, dc = DIRS[d]
        if player == -1:
            r, c, dr, dc = E.ROWS - 1 - r, E.COLS - 1 - c, -dr, -dc
        for m in legal:
            if m.frm == (r, c) and (np.sign(m.to[0] - r), np.sign(m.to[1] - c)) == (dr, dc):
                return m
        return None
