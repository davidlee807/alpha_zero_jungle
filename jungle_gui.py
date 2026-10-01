"""香港鬥獸棋 (Hong Kong Jungle) — pygame board.

The board and the piece pictures come from your board photo: the program blanks
the 16 starting squares to make an empty board, and cuts each piece picture out
of its starting square.

    pip install pygame
    python jungle_gui.py                      # looks for board.png (or *鬥獸棋*.png) here
    python jungle_gui.py path/to/board.png
    python jungle_gui.py --ai top             # play 西九 against a trained AlphaZero agent
    python jungle_gui.py --ai bottom --model temp_jungle/best.pth.tar --sims 200

Mouse: click a piece, then a highlighted square.
Keys:  arrows + Enter to play, Esc cancel, U undo, N new game.
"""
from __future__ import annotations

import glob
import os
import sys
import threading
import time
from dataclasses import replace

import pygame

import jungle_engine as J

HERE = os.path.dirname(os.path.abspath(__file__))

# Grid lines (first, last pixel) measured on the 1896 x 2424 board photo.
# If your copy has another resolution the positions are scaled to fit.
REF_W, REF_H = 1896, 2424
REF_COL_LINES = [(22, 29), (286, 294), (550, 558), (811, 819), (1075, 1083),
                 (1337, 1345), (1601, 1609), (1866, 1874)]
REF_ROW_LINES = [(20, 27), (284, 291), (548, 555), (812, 819), (1074, 1084),
                 (1338, 1348), (1603, 1611), (1866, 1874), (2130, 2137), (2395, 2402)]
SPRITE_INSET = 7          # trimmed from each cut-out so no grid line comes along

INK = (216, 38, 28)
INK_DEEP = (150, 22, 14)
BLUE_INK = (30, 70, 160)
TEXT = (58, 20, 16)
MUTED = (134, 84, 78)
PAGE = (255, 255, 255)
PANEL_W = 340
MARGIN = 20
ANIM_SECS = 0.18

FONT_NAMES = [
    "microsoftjhenghei", "microsoftjhengheiui", "pingfanghk", "pingfangtc", "pingfang",
    "heititc", "notosanscjktc", "notosanscjkhk", "notosanscjk", "notoserifcjktc",
    "sourcehansanstc", "wenquanyizenhei", "wenquanyimicrohei", "microsoftyahei",
    "simhei", "mingliu", "pmingliu", "arialunicodems",
]
FONT_FILES = [
    "C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/mingliu.ttc",
    "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc", "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def find_font_path():
    local = sorted(glob.glob(os.path.join(HERE, "*.tt[fc]")) + glob.glob(os.path.join(HERE, "*.otf")))
    if local:
        return local[0]
    for name in FONT_NAMES:
        path = pygame.font.match_font(name)
        if path:
            return path
    for path in FONT_FILES:
        if os.path.exists(path):
            return path
    return None


def find_board_image(given=None):
    if given:
        return given
    for pattern in ("board.png", "*鬥獸棋*.png", "*.png"):
        hits = sorted(glob.glob(os.path.join(HERE, pattern)))
        if hits:
            return hits[0]
    return None


def side_label(side):
    return f"{J.SIDE_NAMES[side][0]} ({'bottom' if side == J.SOUTH else 'top'})"


class BoardArt:
    """Empty board and piece cut-outs made from the board photo."""

    def __init__(self, path):
        src = pygame.image.load(path).convert()
        self.w, self.h = src.get_size()
        sx, sy = self.w / REF_W, self.h / REF_H
        self.col_lines = [(a * sx, b * sx) for a, b in REF_COL_LINES]
        self.row_lines = [(a * sy, b * sy) for a, b in REF_ROW_LINES]
        self.inset = SPRITE_INSET * sx

        self.empty = src.copy()
        self.sprites = {}       # (side, rank, blue) -> Surface with transparency
        ins = round(self.inset)
        for side, places in J.START.items():
            for rank, (r, c) in places.items():
                cell = self.interior(r, c)
                # to_surface() builds a 32-bit surface with alpha, so the source
                # tile must be in the same format.
                crop = src.subsurface(cell.inflate(-2 * ins, -2 * ins)).convert_alpha()
                paper = pygame.mask.from_threshold(crop, (255, 255, 255, 255), (70, 70, 70, 255))
                paper.invert()                                  # now: the red ink
                self.sprites[(side, rank, False)] = paper.to_surface(
                    setsurface=crop, unsetcolor=(0, 0, 0, 0))
                self.sprites[(side, rank, True)] = paper.to_surface(
                    setcolor=BLUE_INK + (255,), unsetcolor=(0, 0, 0, 0))
                self.empty.fill(PAGE, cell)
        self.k = None
        self.board_s = None
        self.sprites_s = {}

    def interior_f(self, r, c):
        x0 = self.col_lines[c][1] + 1
        x1 = self.col_lines[c + 1][0]
        y0 = self.row_lines[r][1] + 1
        y1 = self.row_lines[r + 1][0]
        return x0, y0, x1, y1

    def interior(self, r, c):
        x0, y0, x1, y1 = self.interior_f(r, c)
        return pygame.Rect(round(x0), round(y0), round(x1 - x0), round(y1 - y0))

    def rescale(self, k):
        if k == self.k:
            return
        self.k = k
        self.board_s = pygame.transform.smoothscale(
            self.empty, (max(1, round(self.w * k)), max(1, round(self.h * k))))
        self.sprites_s = {
            key: pygame.transform.smoothscale(
                s, (max(1, round(s.get_width() * k)), max(1, round(s.get_height() * k))))
            for key, s in self.sprites.items()
        }


class App:
    def __init__(self, image_path, ai_side=None, bot=None):
        self.ai_side = ai_side
        self.bot = bot
        self.thinking = None        # (thread, state it is thinking about, result box)
        pygame.init()
        pygame.display.set_caption("香港鬥獸棋")
        info = pygame.display.Info()
        bh = max(480, min(info.current_h - 140, 1000))
        bw = round(bh * REF_W / REF_H)
        self.screen = pygame.display.set_mode((bw + PANEL_W + 3 * MARGIN, bh + 2 * MARGIN),
                                              pygame.RESIZABLE)
        self.art = BoardArt(image_path)
        self.font_path = find_font_path()
        if self.font_path is None:
            print("No Chinese font found; Chinese text may show as boxes. "
                  "Put a .ttf/.ttc/.otf font with Chinese characters next to this script.")
        self.fonts = {}
        self.rules = J.Rules()
        self.blue_top = False
        self.clock = pygame.time.Clock()
        self.hover = None
        self.cursor = (8, 3)
        self.keyboard = False
        self.hits = []
        self.new_game()
        self.layout()

    # ---------- game actions ----------
    def new_game(self):
        self.state = J.initial_state(self.rules)
        self.history = []
        self.log = []
        self.anim = None
        self.clear_selection()
        if self.bot is not None:
            self.bot.reset()

    def clear_selection(self):
        self.sel = None
        self.targets = []

    def undo(self):
        if not self.history:
            return
        self._pop_move()
        # Against the agent, take back its reply too so it is your turn again.
        if self.ai_side is not None and self.state.turn == self.ai_side and self.history:
            self._pop_move()

    def _pop_move(self):
        self.state = self.history.pop()
        self.log.pop()
        self.anim = None
        self.clear_selection()

    def ai_to_move(self):
        return (self.bot is not None and self.state.winner is None
                and self.state.turn == self.ai_side)

    def ai_step(self):
        """Run the agent in a background thread so the window stays responsive."""
        if self.thinking is not None:
            thread, snapshot, box = self.thinking
            if thread.is_alive():
                return
            self.thinking = None
            if 'error' in box:
                print("The agent stopped with an error:", box['error'])
                self.bot = None
                return
            if snapshot is self.state and box['move'] in J.legal_moves(self.state, self.rules):
                self.play(box['move'])
            return
        if not self.ai_to_move() or self.anim is not None:
            return
        snapshot, log = self.state, list(self.log)
        legal = J.legal_moves(snapshot, self.rules)
        box = {}

        def work():
            try:
                box['move'] = self.bot.choose(snapshot, log, legal)
            except Exception as exc:          # shown in the console, the game carries on
                box['error'] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self.thinking = (thread, snapshot, box)

    def play(self, move):
        self.history.append(self.state)
        self.log.append(move)
        self.state = J.apply_move(self.state, move, self.rules)
        self.anim = (move, time.perf_counter())
        self.clear_selection()

    def click_cell(self, rc):
        if rc is None or self.state.winner is not None or self.ai_to_move():
            return
        move = next((m for m in self.targets if m.to == rc), None)
        if move:
            self.play(move)
            return
        p = self.state.at(*rc)
        if p is not None and p.side == self.state.turn and self.sel != rc:
            self.sel = rc
            self.targets = J.moves_from(self.state, rc[0], rc[1], self.rules)
        else:
            self.clear_selection()

    def toggle_rule(self, attr):
        self.rules = replace(self.rules, **{attr: not getattr(self.rules, attr)})
        if self.sel:
            self.targets = J.moves_from(self.state, self.sel[0], self.sel[1], self.rules)

    def toggle_first(self):
        first = J.NORTH if self.rules.first == J.SOUTH else J.SOUTH
        self.rules = replace(self.rules, first=first)
        if not self.log:
            self.new_game()

    def toggle_blue(self):
        self.blue_top = not self.blue_top

    # ---------- geometry ----------
    def layout(self):
        self.screen = pygame.display.get_surface()
        W, H = self.screen.get_size()
        k = min((H - 2 * MARGIN) / self.art.h, (W - PANEL_W - 3 * MARGIN) / self.art.w)
        self.k = max(k, 0.05)
        self.art.rescale(self.k)
        bw, bh = self.art.board_s.get_size()
        self.bx = MARGIN
        self.by = max(MARGIN, (H - bh) // 2)
        self.px = self.bx + bw + MARGIN
        self.pw = max(200, W - self.px - MARGIN)

    def cell_rect(self, r, c):
        x0, y0, x1, y1 = self.art.interior_f(r, c)
        k = self.k
        return pygame.Rect(round(self.bx + x0 * k), round(self.by + y0 * k),
                           round((x1 - x0) * k), round((y1 - y0) * k))

    def cell_at(self, pos):
        x = (pos[0] - self.bx) / self.k
        y = (pos[1] - self.by) / self.k
        cols, rows = self.art.col_lines, self.art.row_lines
        if not (cols[0][0] <= x < cols[-1][1] and rows[0][0] <= y < rows[-1][1]):
            return None
        c = sum(1 for i in range(1, J.COLS) if x >= cols[i][0])
        r = sum(1 for i in range(1, J.ROWS) if y >= rows[i][0])
        return (r, c)

    # ---------- drawing helpers ----------
    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            f = pygame.font.Font(self.font_path, size)
            f.set_bold(bold)
            self.fonts[key] = f
        return self.fonts[key]

    def text(self, s, size, color=TEXT, bold=False):
        return self.font(size, bold).render(s, True, color)

    def sprite(self, piece):
        blue = self.blue_top and piece.side == J.NORTH
        return self.art.sprites_s[(piece.side, piece.rank, blue)]

    def thumb(self, piece, size):
        s = self.sprite(piece)
        return pygame.transform.smoothscale(s, (size, size))

    def draw_piece(self, piece, rect, plate=False, dim=False):
        if plate:
            p = pygame.Surface(rect.size, pygame.SRCALPHA)
            p.fill((255, 255, 255, 225))
            self.screen.blit(p, rect.topleft)
        spr = self.sprite(piece)
        if dim:
            spr = spr.copy()
            spr.set_alpha(115)
        off = round(self.art.inset * self.k)
        self.screen.blit(spr, (rect.x + off, rect.y + off))

    # ---------- drawing ----------
    def draw(self):
        self.screen.fill(PAGE)
        self.draw_board()
        self.draw_panel()

    def draw_board(self):
        scr = self.screen
        scr.blit(self.art.board_s, (self.bx, self.by))

        tint = pygame.Surface(scr.get_size(), pygame.SRCALPHA)
        if self.log:
            last = self.log[-1]
            for rc in (last.frm, last.to):
                tint.fill((216, 38, 28, 34), self.cell_rect(*rc))
        if self.sel:
            tint.fill((216, 38, 28, 80), self.cell_rect(*self.sel))
        scr.blit(tint, (0, 0))

        moving_to, progress = None, 1.0
        if self.anim:
            move, t0 = self.anim
            progress = (time.perf_counter() - t0) / ANIM_SECS
            if progress >= 1:
                self.anim = None
            else:
                moving_to = move.to

        for r in range(J.ROWS):
            for c in range(J.COLS):
                p = self.state.at(r, c)
                if p is None or (r, c) == moving_to:
                    continue
                special = (r, c) in J.TRAPS or (r, c) in J.DENS or J.is_river(r, c)
                trap = J.TRAPS.get((r, c))
                self.draw_piece(p, self.cell_rect(r, c), plate=special,
                                dim=bool(trap and trap[0] != p.side))

        if moving_to:
            move = self.anim[0]
            a, b = self.cell_rect(*move.frm), self.cell_rect(*move.to)
            e = 1 - (1 - progress) ** 3
            rect = pygame.Rect(round(a.x + (b.x - a.x) * e), round(a.y + (b.y - a.y) * e), a.w, a.h)
            self.draw_piece(move.piece, rect)

        marks = pygame.Surface(scr.get_size(), pygame.SRCALPHA)
        for m in self.targets:
            rect = self.cell_rect(*m.to)
            if m.capture:
                pygame.draw.circle(marks, INK_DEEP + (220,), rect.center,
                                   int(min(rect.w, rect.h) * 0.47), max(3, rect.w // 22))
            else:
                pygame.draw.circle(marks, INK + (150,), rect.center, max(4, rect.w // 10))
        scr.blit(marks, (0, 0))

        if self.keyboard:
            pygame.draw.rect(scr, (42, 13, 10), self.cell_rect(*self.cursor), 3)
        elif self.hover and self.state.winner is None:
            p = self.state.at(*self.hover)
            if (p and p.side == self.state.turn) or any(m.to == self.hover for m in self.targets):
                pygame.draw.rect(scr, INK, self.cell_rect(*self.hover), 2)

        if self.state.winner is not None:
            self.draw_result()

    def draw_result(self):
        scr = self.screen
        bw, bh = self.art.board_s.get_size()
        veil = pygame.Surface((bw, bh), pygame.SRCALPHA)
        veil.fill((255, 255, 255, 190))
        scr.blit(veil, (self.bx, self.by))

        st = self.state
        if st.reason == "den":
            winner_piece = self.log[-1].piece.name
            detail = f"{winner_piece} entered {J.DENS[self.log[-1].to][1]}."
        else:
            detail = f"{side_label(1 - st.winner)} has no legal moves."
        lines = [self.text(f"{side_label(st.winner)} wins", 34, INK, bold=True),
                 self.text(detail, 20),
                 self.text("Press N or click New game.", 17, MUTED)]
        w = max(l.get_width() for l in lines) + 60
        h = sum(l.get_height() for l in lines) + 24 + 56
        card = pygame.Rect(0, 0, w, h)
        card.center = (self.bx + bw // 2, self.by + bh // 2)
        pygame.draw.rect(scr, PAGE, card)
        pygame.draw.rect(scr, INK, card, 4)
        pygame.draw.rect(scr, INK, card.inflate(12, 12), 2)
        y = card.y + 28
        for l in lines:
            scr.blit(l, (card.centerx - l.get_width() // 2, y))
            y += l.get_height() + 12

    def button(self, x, y, label, action, primary=True, enabled=True):
        t = self.text(label, 18, PAGE if primary else INK, bold=True)
        rect = pygame.Rect(x, y, t.get_width() + 32, t.get_height() + 16)
        if primary:
            pygame.draw.rect(self.screen, INK, rect, border_radius=3)
        else:
            pygame.draw.rect(self.screen, INK, rect, 2, border_radius=3)
        if not enabled:
            t.set_alpha(90)
        self.screen.blit(t, (rect.x + 16, rect.y + 8))
        if enabled:
            self.hits.append((rect, action))
        return rect

    def checkbox(self, x, y, label, checked, action):
        box = pygame.Rect(x, y + 3, 18, 18)
        pygame.draw.rect(self.screen, INK, box, 2)
        if checked:
            pygame.draw.lines(self.screen, INK, False,
                              [(box.x + 4, box.y + 9), (box.x + 8, box.y + 13), (box.x + 14, box.y + 4)], 3)
        t = self.text(label, 16)
        self.screen.blit(t, (x + 28, y))
        self.hits.append((pygame.Rect(x, y, 28 + t.get_width(), max(24, t.get_height())), action))
        return y + max(26, t.get_height() + 6)

    def draw_panel(self):
        scr = self.screen
        self.hits = []
        x, y, w = self.px, self.by, self.pw
        bottom = self.by + self.art.board_s.get_height()

        t = self.text("香港鬥獸棋", 42, INK, bold=True)
        scr.blit(t, (x, y))
        y += t.get_height() + 2
        if self.bot is not None:
            sub = f"You play {J.SIDE_NAMES[1 - self.ai_side][0]} against the AlphaZero agent"
        else:
            sub = "Hong Kong Jungle, two players on one screen"
        t = self.text(sub, 15, MUTED)
        scr.blit(t, (x, y))
        y += t.get_height() + 18

        side = self.state.winner if self.state.winner is not None else self.state.turn
        scr.blit(self.thumb(J.Piece(side, J.ELEPHANT), 44), (x, y))
        if self.state.winner is not None:
            msg = f"{side_label(side)} wins"
        elif self.ai_to_move():
            msg = f"{side_label(side)} is thinking…"
        else:
            msg = f"{side_label(side)} to move"
        t = self.text(msg, 22, TEXT, bold=True)
        scr.blit(t, (x + 54, y + 22 - t.get_height() // 2))
        y += 58

        r1 = self.button(x, y, "New game", self.new_game)
        self.button(r1.right + 10, y, "Undo move", self.undo, primary=False, enabled=bool(self.history))
        y = r1.bottom + 20

        y = self.heading(x, y, w, "Captured")
        taken = {J.SOUTH: [], J.NORTH: []}
        for m in self.log:
            if m.capture:
                taken[m.piece.side].append(m.capture)
        for s in (J.SOUTH, J.NORTH):
            t = self.text(f"{J.SIDE_NAMES[s][0]} took", 16, MUTED)
            scr.blit(t, (x, y + 8))
            tx = x + 78
            pieces = sorted(taken[s], key=lambda p: -p.rank)
            if not pieces:
                scr.blit(self.text("none yet", 16, MUTED), (tx, y + 8))
            for p in pieces:
                if tx + 34 > x + w:
                    break
                scr.blit(self.thumb(p, 34), (tx, y))
                tx += 36
            y += 38
        y += 10

        y = self.heading(x, y, w, "House rules")
        y = self.checkbox(x, y, "狸超人 jumps rivers sideways", self.rules.lion_side_jump,
                          lambda: self.toggle_rule("lion_side_jump"))
        y = self.checkbox(x, y, "中鸞扮 jumps rivers sideways", self.rules.tiger_side_jump,
                          lambda: self.toggle_rule("tiger_side_jump"))
        y = self.checkbox(x, y, "Capture between river and bank", self.rules.shore_capture,
                          lambda: self.toggle_rule("shore_capture"))
        y = self.checkbox(x, y, "中央正虎 can take 蟻民", self.rules.elephant_eats_rat,
                          lambda: self.toggle_rule("elephant_eats_rat"))
        y = self.checkbox(x, y, "中環 moves first", self.rules.first == J.NORTH, self.toggle_first)
        y = self.checkbox(x, y, "Top side in blue ink", self.blue_top, self.toggle_blue)
        y += 12

        help_lines = ["Click a piece, then a marked square.",
                      "Keys: U undo, N new game, Esc cancel,", "arrows and Enter to play."]
        help_h = len(help_lines) * 20
        y = self.heading(x, y, w, "Moves")
        line_h = 22
        room = max(0, (bottom - help_h - 12 - y) // line_h)
        if not self.log:
            scr.blit(self.text("No moves yet.", 16, MUTED), (x, y))
        start = max(0, len(self.log) - room)
        for i in range(start, len(self.log)):
            m = self.log[i]
            t = self.text(f"{i + 1}. {J.SIDE_NAMES[m.piece.side][0]}  {J.notation(m)}", 16)
            scr.blit(t, (x, y))
            y += line_h

        hy = bottom - help_h
        for line in help_lines:
            scr.blit(self.text(line, 14, MUTED), (x, hy))
            hy += 20

    def heading(self, x, y, w, label):
        t = self.text(label, 18, TEXT, bold=True)
        self.screen.blit(t, (x, y))
        y += t.get_height() + 4
        pygame.draw.line(self.screen, INK, (x, y), (x + w, y), 2)
        return y + 8

    # ---------- main loop ----------
    def handle_key(self, key):
        r, c = self.cursor
        if key == pygame.K_UP:
            r = max(0, r - 1)
        elif key == pygame.K_DOWN:
            r = min(J.ROWS - 1, r + 1)
        elif key == pygame.K_LEFT:
            c = max(0, c - 1)
        elif key == pygame.K_RIGHT:
            c = min(J.COLS - 1, c + 1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self.keyboard = True
            self.click_cell(self.cursor)
            return
        elif key == pygame.K_ESCAPE:
            self.clear_selection()
            return
        elif key == pygame.K_u:
            self.undo()
            return
        elif key == pygame.K_n:
            self.new_game()
            return
        else:
            return
        self.keyboard = True
        self.cursor = (r, c)

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return
                if e.type == pygame.VIDEORESIZE:
                    self.layout()
                elif e.type == pygame.MOUSEMOTION:
                    self.hover = self.cell_at(e.pos)
                    self.keyboard = False
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for rect, action in self.hits:
                        if rect.collidepoint(e.pos):
                            action()
                            break
                    else:
                        self.click_cell(self.cell_at(e.pos))
                elif e.type == pygame.KEYDOWN:
                    self.handle_key(e.key)
            self.ai_step()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="香港鬥獸棋 board")
    ap.add_argument("image", nargs="?", help="board photo (default: board.png next to this script)")
    ap.add_argument("--ai", choices=["top", "bottom"],
                    help="let the trained AlphaZero agent play this side")
    ap.add_argument("--model", default=os.path.join("temp_jungle", "best.pth.tar"),
                    help="checkpoint saved by main_jungle.py")
    ap.add_argument("--sims", type=int, default=100, help="MCTS simulations per agent move")
    a = ap.parse_args()

    path = find_board_image(a.image)
    if not path or not os.path.exists(path):
        print("Board photo not found. Put it next to this script as board.png, or run:\n"
              "    python jungle_gui.py path/to/board.png")
        sys.exit(1)

    bot, ai_side = None, None
    if a.ai:
        from az_bot import AlphaZeroBot     # needs PyTorch and the alpha-zero-general files
        bot = AlphaZeroBot(a.model, sims=a.sims)
        ai_side = J.NORTH if a.ai == "top" else J.SOUTH
    App(path, ai_side, bot).run()


if __name__ == "__main__":
    main()
