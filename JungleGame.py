import sys

import numpy as np

sys.path.append('..')
from Game import Game
from .JungleLogic import (ACTION_SIZE, BOTTOM_DEN, BOTTOM_TRAPS, COLS, MIRROR_ACTIONS, RIVER,
                          ROWS, TOP_DEN, TOP_TRAPS, apply_action, board_to_text, den_winner,
                          flip, has_move, initial_board, mirror, valid_moves)

NUM_PLANES = 23


def _const_plane(squares):
    p = np.zeros((ROWS, COLS), dtype=np.float32)
    for r, c in squares:
        p[r, c] = 1.0
    return p


class JungleGame(Game):
    """香港鬥獸棋 with the default house rules, for alpha-zero-general.

    Player 1 is 西九 (bottom, moves first); player -1 is 中環 (top).
    In the board given to the network (the canonical form) the player to move
    always has the positive pieces at the bottom.

    Self-play needs every game to finish, so two limits are added on top of the
    rules; reaching either one is a draw:
      no_capture_limit  plies in a row without a capture
      max_plies         plies in the whole game
    """

    DRAW = 1e-4

    def __init__(self, no_capture_limit=100, max_plies=300):
        super().__init__()
        self.no_capture_limit = no_capture_limit
        self.max_plies = max_plies
        self._terrain = np.stack([
            _const_plane(RIVER),
            _const_plane(BOTTOM_TRAPS),
            _const_plane(TOP_TRAPS),
            _const_plane([BOTTOM_DEN]),
            _const_plane([TOP_DEN]),
        ])

    # ----- Game interface -----
    def getInitBoard(self):
        return initial_board()

    def getBoardSize(self):
        return (ROWS, COLS)

    def getActionSize(self):
        return ACTION_SIZE

    def getNextState(self, board, player, action):
        # Actions are always given from the mover's point of view (canonical form).
        canonical = self.getCanonicalForm(board, player)
        after = apply_action(canonical, action)
        return self.getCanonicalForm(after, player), -player

    def getValidMoves(self, board, player):
        return valid_moves(self.getCanonicalForm(board, player))

    def getGameEnded(self, board, player):
        # `player` is the player to move; positive pieces belong to player 1 of this board.
        w = den_winner(board)
        if w == 0 and not has_move(self.getCanonicalForm(board, player)):
            w = -player
        if w != 0:
            return 1 if w == player else -1
        if board[ROWS, 0] >= self.no_capture_limit or board[ROWS, 1] >= self.max_plies:
            return self.DRAW
        return 0

    def getCanonicalForm(self, board, player):
        return board if player == 1 else flip(board)

    def getSymmetries(self, board, pi):
        pi = np.asarray(pi, dtype=np.float32)
        return [(board, pi), (mirror(board), pi[MIRROR_ACTIONS])]

    def stringRepresentation(self, board):
        return board.tobytes()

    # ----- network input -----
    def encode(self, boards):
        """(N, 10, 7) canonical boards -> (N, 23, 9, 7) float32 planes:
        8 own pieces by rank, 8 enemy pieces by rank, river, own traps, enemy traps,
        own den, enemy den, no-capture counter, ply counter."""
        boards = np.asarray(boards)
        n = boards.shape[0]
        grid = boards[:, :ROWS, :].astype(np.int16)
        ranks = np.arange(1, 9, dtype=np.int16)[None, :, None, None]
        own = (grid[:, None] == ranks).astype(np.float32)
        enemy = (grid[:, None] == -ranks).astype(np.float32)
        terrain = np.broadcast_to(self._terrain, (n,) + self._terrain.shape)
        nc = (boards[:, ROWS, 0].astype(np.float32) / self.no_capture_limit)[:, None, None, None]
        ply = (boards[:, ROWS, 1].astype(np.float32) / self.max_plies)[:, None, None, None]
        counters = np.concatenate([np.broadcast_to(nc, (n, 1, ROWS, COLS)),
                                   np.broadcast_to(ply, (n, 1, ROWS, COLS))], axis=1)
        return np.ascontiguousarray(np.concatenate([own, enemy, terrain, counters], axis=1),
                                    dtype=np.float32)

    @staticmethod
    def display(board):
        print(board_to_text(board))
        print("  uppercase = 西九 (bottom), lowercase = 中環 (top); "
              "E 中央正虎 L 狸超人 T 中鸞扮 P 金融大鱷 W 突首龜 D 狗官 C 耳猿 R 蟻民")
