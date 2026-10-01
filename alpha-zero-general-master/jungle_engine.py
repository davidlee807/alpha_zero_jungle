"""Rules engine for 香港鬥獸棋 (Hong Kong Jungle). No pygame needed.

Coordinates are (r, c): r = 0..8 from top to bottom, c = 0..6 from left to right.
Side 0 (SOUTH) is 西九 and starts at the bottom; side 1 (NORTH) is 中環 at the top.

States are immutable and hashable, so they can be used directly as dictionary
keys in search code (transposition tables, abstraction maps, heuristics...).

    state = initial_state()
    moves = legal_moves(state)
    state = apply_move(state, moves[0])
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, List, NamedTuple, Optional, Tuple

ROWS, COLS = 9, 7
SOUTH, NORTH = 0, 1
FILES = "abcdefg"

ELEPHANT, LION, TIGER, LEOPARD, WOLF, DOG, CAT, RAT = 8, 7, 6, 5, 4, 3, 2, 1

# rank -> (name, classic Jungle piece it replaces)
KINDS: Dict[int, Tuple[str, str]] = {
    8: ("中央正虎", "elephant 象"),
    7: ("狸超人", "lion 獅"),
    6: ("中鸞扮", "tiger 虎"),
    5: ("金融大鱷", "leopard 豹"),
    4: ("突首龜", "wolf 狼"),
    3: ("狗官", "dog 狗"),
    2: ("耳猿", "cat 貓"),
    1: ("蟻民", "rat 鼠"),
}
SIDE_NAMES = {SOUTH: ("西九", "West Kowloon"), NORTH: ("中環", "Central")}

# square -> (owner, name)
TRAPS = {
    (8, 2): (SOUTH, "高鐵"), (7, 3): (SOUTH, "嶺匯"), (8, 4): (SOUTH, "劏房"),
    (0, 4): (NORTH, "累計期權"), (1, 3): (NORTH, "強積金"), (0, 2): (NORTH, "迷你債券"),
}
DENS = {(8, 3): (SOUTH, "西九"), (0, 3): (NORTH, "中環")}

SWIMMERS = frozenset({LEOPARD, WOLF})          # 金融大鱷, 突首龜
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

START: Dict[int, Dict[int, Tuple[int, int]]] = {
    SOUTH: {ELEPHANT: (6, 0), WOLF: (6, 2), LEOPARD: (6, 4), RAT: (6, 6),
            CAT: (7, 1), DOG: (7, 5), TIGER: (8, 0), LION: (8, 6)},
}
START[NORTH] = {rank: (ROWS - 1 - r, COLS - 1 - c) for rank, (r, c) in START[SOUTH].items()}


class Piece(NamedTuple):
    side: int
    rank: int

    @property
    def name(self) -> str:
        return KINDS[self.rank][0]


class Move(NamedTuple):
    frm: Tuple[int, int]
    to: Tuple[int, int]
    piece: Piece
    capture: Optional[Piece]
    jump: bool


@dataclass(frozen=True)
class Rules:
    """Rule choices the Wikipedia summary leaves open. Defaults follow classic Jungle."""
    lion_side_jump: bool = True      # 狸超人 may jump a river sideways (it always may lengthways)
    tiger_side_jump: bool = True     # 中鸞扮 may jump a river sideways
    shore_capture: bool = False      # a piece may capture between river and bank
    elephant_eats_rat: bool = True   # 中央正虎 may take 蟻民 (蟻民 may always take 中央正虎)
    first: int = SOUTH


DEFAULT_RULES = Rules()


@dataclass(frozen=True)
class State:
    board: Tuple[Optional[Piece], ...]   # ROWS*COLS entries, row-major
    turn: int
    winner: Optional[int] = None
    reason: Optional[str] = None         # "den" or "stuck"

    def at(self, r: int, c: int) -> Optional[Piece]:
        return self.board[r * COLS + c]


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS


def is_river(r: int, c: int) -> bool:
    return 3 <= r <= 5 and c in (1, 2, 4, 5)


def square_name(r: int, c: int) -> str:
    return f"{FILES[c]}{ROWS - r}"


def initial_state(rules: Rules = DEFAULT_RULES) -> State:
    board: List[Optional[Piece]] = [None] * (ROWS * COLS)
    for side, places in START.items():
        for rank, (r, c) in places.items():
            board[r * COLS + c] = Piece(side, rank)
    return State(tuple(board), rules.first)


def can_jump(rank: int, vertical: bool, rules: Rules) -> bool:
    if rank == LION:
        return vertical or rules.lion_side_jump
    if rank == TIGER:
        return vertical or rules.tiger_side_jump
    return False


def can_capture(att: Piece, frm: Tuple[int, int], dfd: Piece, to: Tuple[int, int],
                rules: Rules = DEFAULT_RULES) -> bool:
    trap = TRAPS.get(to)
    if trap and trap[0] == att.side:          # enemy standing in my trap: anyone takes it
        return True
    if not rules.shore_capture and is_river(*frm) != is_river(*to):
        return False
    if att.rank == RAT and dfd.rank == ELEPHANT:
        return True
    if att.rank == ELEPHANT and dfd.rank == RAT:
        return rules.elephant_eats_rat
    return att.rank >= dfd.rank


def moves_from(state: State, r: int, c: int, rules: Rules = DEFAULT_RULES) -> List[Move]:
    p = state.at(r, c)
    if p is None:
        return []
    out: List[Move] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc):
            continue
        jump = False
        if is_river(nr, nc) and p.rank not in SWIMMERS:
            if not can_jump(p.rank, dr != 0, rules):
                continue
            blocked = False
            while in_bounds(nr, nc) and is_river(nr, nc):
                if state.at(nr, nc) is not None:
                    blocked = True
                    break
                nr, nc = nr + dr, nc + dc
            if blocked or not in_bounds(nr, nc):
                continue
            jump = True
        den = DENS.get((nr, nc))
        if den and den[0] == p.side:
            continue
        target = state.at(nr, nc)
        if target is not None:
            if target.side == p.side or not can_capture(p, (r, c), target, (nr, nc), rules):
                continue
        out.append(Move((r, c), (nr, nc), p, target, jump))
    return out


def legal_moves(state: State, rules: Rules = DEFAULT_RULES) -> List[Move]:
    if state.winner is not None:
        return []
    moves: List[Move] = []
    for i, p in enumerate(state.board):
        if p is not None and p.side == state.turn:
            moves.extend(moves_from(state, i // COLS, i % COLS, rules))
    return moves


def apply_move(state: State, m: Move, rules: Rules = DEFAULT_RULES) -> State:
    board = list(state.board)
    fi, ti = m.frm[0] * COLS + m.frm[1], m.to[0] * COLS + m.to[1]
    piece = board[fi]
    board[fi], board[ti] = None, piece
    nxt = State(tuple(board), 1 - state.turn)
    den = DENS.get(m.to)
    if den and den[0] != piece.side:
        return replace(nxt, winner=piece.side, reason="den")
    if not legal_moves(nxt, rules):
        return replace(nxt, winner=piece.side, reason="stuck")
    return nxt


def notation(m: Move) -> str:
    s = f"{m.piece.name} {square_name(*m.frm)}{'×' if m.capture else '–'}{square_name(*m.to)}"
    return s + (f" {m.capture.name}" if m.capture else "")


if __name__ == "__main__":
    # Quick self-check: opening move count, jumps, and random games.
    s0 = initial_state()
    assert len(legal_moves(s0)) == 26, len(legal_moves(s0))

    empty = State(tuple([None] * (ROWS * COLS)), SOUTH)
    b = list(empty.board)
    b[2 * COLS + 1] = Piece(SOUTH, LION)
    b[8 * COLS + 0] = Piece(NORTH, RAT)
    s = State(tuple(b), SOUTH)
    assert (6, 1) in [m.to for m in moves_from(s, 2, 1)]          # jumps lengthways
    b[4 * COLS + 1] = Piece(NORTH, LEOPARD)
    s = State(tuple(b), SOUTH)
    assert (6, 1) not in [m.to for m in moves_from(s, 2, 1)]      # blocked by a swimmer

    rng = random.Random(1)
    results = {SOUTH: 0, NORTH: 0, None: 0}
    for _ in range(500):
        st = initial_state()
        for _ in range(400):
            if st.winner is not None:
                break
            st = apply_move(st, rng.choice(legal_moves(st)))
        results[st.winner] += 1
    print("self-check passed; random games won by 西九/中環/unfinished:",
          results[SOUTH], results[NORTH], results[None])
