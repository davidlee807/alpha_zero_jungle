import numpy as np

from .JungleLogic import COLS, ROWS, TOP_DEN, board_to_text, destination


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        return int(np.random.choice(np.flatnonzero(valids)))


class GreedyJunglePlayer:
    """Wins at once if it can, otherwise takes the biggest piece it can,
    otherwise moves closest to the enemy den. Useful as a sparring baseline."""

    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = np.flatnonzero(self.game.getValidMoves(board, 1))
        grid = board[:ROWS].tolist()
        best, best_score = [], None
        for a in valids:
            sq, d = divmod(int(a), 4)
            r, c = divmod(sq, COLS)
            nr, nc = destination(grid, r, c, d)
            if (nr, nc) == TOP_DEN:
                return int(a)
            taken = -grid[nr][nc] if grid[nr][nc] < 0 else 0
            dist = abs(nr - TOP_DEN[0]) + abs(nc - TOP_DEN[1])
            score = (taken, -dist)
            if best_score is None or score > best_score:
                best, best_score = [a], score
            elif score == best_score:
                best.append(a)
        return int(np.random.choice(best))


class HumanJunglePlayer:
    """Type moves as two squares, e.g. `a3 a4`. The board is shown from your side:
    your pieces are uppercase and you play upwards."""

    def __init__(self, game):
        self.game = game

    def play(self, board):
        print(board_to_text(board))
        valids = self.game.getValidMoves(board, 1)
        grid = board[:ROWS].tolist()
        options = {}
        for a in np.flatnonzero(valids):
            sq, d = divmod(int(a), 4)
            r, c = divmod(sq, COLS)
            nr, nc = destination(grid, r, c, d)
            key = f"{'abcdefg'[c]}{ROWS - r} {'abcdefg'[nc]}{ROWS - nr}"
            options[key] = int(a)
        print("Your moves:", ", ".join(sorted(options)))
        while True:
            text = " ".join(input("Your move: ").lower().split())
            if text in options:
                return options[text]
            print("Not a legal move. Type it like `a3 a4`.")
