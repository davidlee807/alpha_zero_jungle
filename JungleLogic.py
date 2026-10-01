"""Board logic for 香港鬥獸棋 with the default house rules, on a signed integer board.

Board: numpy int16 array, shape (10, 7).
  rows 0..8  the squares. +rank for one side's pieces, -rank for the other's.
             Positive pieces always play from the BOTTOM: their den is (8, 3)
             and their traps are (8, 2), (7, 3), (8, 4).
  row 9      counters: [9, 0] = plies since the last capture, [9, 1] = plies played.

Ranks: 8 中央正虎, 7 狸超人, 6 中鸞扮, 5 金融大鱷, 4 突首龜, 3 狗官, 2 耳猿, 1 蟻民.

Default rules (the GUI's default house rules):
  * only 金融大鱷 and 突首龜 enter the rivers;
  * 狸超人 and 中鸞扮 jump a river lengthways or sideways; any piece in the river blocks;
  * no capturing between river and bank;
  * higher-or-equal rank captures; 蟻民 takes 中央正虎 and 中央正虎 takes 蟻民;
  * any piece takes an enemy standing in its own side's trap;
  * no piece enters its own den; entering the enemy den wins;
  * a player with no legal move loses.

Actions: action = square * 4 + direction, square = r * 7 + c,
direction 0 up, 1 down, 2 left, 3 right (a jump uses the same direction).
"""
import numpy as np

ROWS, COLS = 9, 7
NUM_SQUARES = ROWS * COLS
ACTION_SIZE = NUM_SQUARES * 4
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

ELEPHANT, LION, TIGER, LEOPARD, WOLF, DOG, CAT, RAT = 8, 7, 6, 5, 4, 3, 2, 1
SWIMMERS = frozenset({LEOPARD, WOLF})
JUMPERS = frozenset({LION, TIGER})

RIVER = frozenset((r, c) for r in (3, 4, 5) for c in (1, 2, 4, 5))
BOTTOM_TRAPS = frozenset({(8, 2), (7, 3), (8, 4)})
TOP_TRAPS = frozenset({(0, 2), (1, 3), (0, 4)})
BOTTOM_DEN = (8, 3)
TOP_DEN = (0, 3)

# Starting squares of the bottom side; the top side is the 180-degree rotation.
START_BOTTOM = {ELEPHANT: (6, 0), WOLF: (6, 2), LEOPARD: (6, 4), RAT: (6, 6),
                CAT: (7, 1), DOG: (7, 5), TIGER: (8, 0), LION: (8, 6)}

LETTERS = {ELEPHANT: "E", LION: "L", TIGER: "T", LEOPARD: "P",
           WOLF: "W", DOG: "D", CAT: "C", RAT: "R"}


def initial_board():
    b = np.zeros((ROWS + 1, COLS), dtype=np.int16)
    for rank, (r, c) in START_BOTTOM.items():
        b[r, c] = rank
        b[ROWS - 1 - r, COLS - 1 - c] = -rank
    return b


def flip(board):
    """Rotate 180 degrees and swap colours. The rules are unchanged by this."""
    out = board.copy()
    out[:ROWS] = -board[:ROWS][::-1, ::-1]
    return out


def mirror(board):
    """Left-right mirror. The rules are unchanged by this too."""
    out = board.copy()
    out[:ROWS] = board[:ROWS, ::-1]
    return out


def _mirror_action(a):
    sq, d = divmod(a, 4)
    r, c = divmod(sq, COLS)
    d = (0, 1, 3, 2)[d]
    return (r * COLS + (COLS - 1 - c)) * 4 + d


MIRROR_ACTIONS = np.array([_mirror_action(a) for a in range(ACTION_SIZE)], dtype=np.int64)


def can_capture(side, rank, frm, target_rank, to):
    if to in (BOTTOM_TRAPS if side > 0 else TOP_TRAPS):
        return True
    if (frm in RIVER) != (to in RIVER):
        return False
    if rank == RAT and target_rank == ELEPHANT:
        return True
    if rank == ELEPHANT and target_rank == RAT:
        return True
    return rank >= target_rank


def destination(grid, r, c, d):
    """Where the piece on (r, c) lands moving in direction d, or None if illegal.
    grid is board[:9] as nested lists."""
    v = grid[r][c]
    side = 1 if v > 0 else -1
    rank = v if v > 0 else -v
    dr, dc = DIRS[d]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return None
    if (nr, nc) in RIVER and rank not in SWIMMERS:
        if rank not in JUMPERS:
            return None
        while (nr, nc) in RIVER:            # rivers never touch the board edge
            if grid[nr][nc] != 0:
                return None
            nr, nc = nr + dr, nc + dc
    if (nr, nc) == (BOTTOM_DEN if side > 0 else TOP_DEN):
        return None
    t = grid[nr][nc]
    if t != 0:
        if (t > 0) == (side > 0):
            return None
        if not can_capture(side, rank, (r, c), -t if t < 0 else t, (nr, nc)):
            return None
    return nr, nc


def valid_moves(board):
    """Binary vector of legal actions for the POSITIVE side."""
    grid = board[:ROWS].tolist()
    valids = np.zeros(ACTION_SIZE, dtype=np.int8)
    for r in range(ROWS):
        row = grid[r]
        for c in range(COLS):
            if row[c] > 0:
                base = (r * COLS + c) * 4
                for d in range(4):
                    if destination(grid, r, c, d) is not None:
                        valids[base + d] = 1
    return valids


def has_move(board):
    grid = board[:ROWS].tolist()
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] > 0:
                for d in range(4):
                    if destination(grid, r, c, d) is not None:
                        return True
    return False


def apply_action(board, action):
    """Play a POSITIVE-side action and return the new board."""
    sq, d = divmod(int(action), 4)
    r, c = divmod(sq, COLS)
    grid = board[:ROWS].tolist()
    dest = destination(grid, r, c, d)
    if dest is None:
        raise ValueError(f"illegal action {action}")
    nr, nc = dest
    out = board.copy()
    captured = out[nr, nc] != 0
    out[nr, nc] = out[r, c]
    out[r, c] = 0
    out[ROWS, 0] = 0 if captured else out[ROWS, 0] + 1
    out[ROWS, 1] += 1
    return out


def den_winner(board):
    """+1 if a positive piece stands in the top den, -1 if a negative one in the bottom den."""
    if board[TOP_DEN] > 0:
        return 1
    if board[BOTTOM_DEN] < 0:
        return -1
    return 0


def action_to_text(action):
    sq, d = divmod(int(action), 4)
    r, c = divmod(sq, COLS)
    return f"{'abcdefg'[c]}{ROWS - r}{'^v<>'[d]}"


def board_to_text(board):
    lines = ["    a  b  c  d  e  f  g"]
    for r in range(ROWS):
        cells = []
        for c in range(COLS):
            v = int(board[r, c])
            if v > 0:
                ch = LETTERS[v]
            elif v < 0:
                ch = LETTERS[-v].lower()
            elif (r, c) in RIVER:
                ch = "~"
            elif (r, c) in (TOP_DEN, BOTTOM_DEN):
                ch = "@"
            elif (r, c) in TOP_TRAPS or (r, c) in BOTTOM_TRAPS:
                ch = "#"
            else:
                ch = "."
            cells.append(ch)
        lines.append(f" {ROWS - r}  " + "  ".join(cells))
    lines.append(f"    plies: {int(board[ROWS, 1])}, since last capture: {int(board[ROWS, 0])}")
    return "\n".join(lines)
