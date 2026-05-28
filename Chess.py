"""
chess.py — Self-Learning Chess Engine

Play mode  :  python3 chess.py
Train mode :  python3 chess.py --train     (self-play forever, Ctrl+C to stop)

Move syntax:  e2e4  |  e7e8q (promotion)  |  e1g1 (castle)  |  resign

The engine learns from every self-play game it trains.
Weights are saved to chess_weights.npz and improve over time.
"""

import sys
import time
import random
import os
import pickle
import numpy as np

# PyTorch — used for GPU-accelerated neural network
# Install: pip install torch   (then restart)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# ══════════════════════════════════════════════════════
# ── SETTINGS — edit these to tune the engine ──────────
# ══════════════════════════════════════════════════════
TIME_LIMIT  = 8.0   # seconds AI thinks per move in-game  (↑ stronger, ↓ faster)
TRAIN_DEPTH = 3     # search depth during self-play training
                    #  2 = fast (~60 games/hr)
                    #  3 = balanced (default)
                    #  4 = strong  (~10 games/hr)
# ══════════════════════════════════════════════════════

# ── Unicode pieces ────────────────────────────────────────────────────────────
UNICODE = {
    ('K', 'w'): '♚', ('Q', 'w'): '♛', ('R', 'w'): '♜',
    ('B', 'w'): '♝', ('N', 'w'): '♞', ('P', 'w'): '♟',
    ('K', 'b'): '♔', ('Q', 'b'): '♕', ('R', 'b'): '♖',
    ('B', 'b'): '♗', ('N', 'b'): '♘', ('P', 'b'): '♙',
}

COLS = 'abcdefgh'

# ── Piece values ──────────────────────────────────────────────────────────────
PIECE_VALUE = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}

# ── Piece-square tables (white's perspective, row 0 = rank 8) ─────────────────
PST = {
    'P': [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    'N': [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    'B': [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    'R': [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0,
    ],
    'Q': [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    'K_mid': [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
    'K_end': [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50,
    ],
}

def pst_score(kind, color, r, c, endgame=False):
    """Piece-square table bonus, always from white's perspective."""
    table_key = ('K_end' if endgame else 'K_mid') if kind == 'K' else kind
    table = PST.get(table_key, PST.get(kind, [0]*64))
    idx = r * 8 + c if color == 'w' else (7 - r) * 8 + c
    return table[idx]

# ── Board helpers ─────────────────────────────────────────────────────────────
def rc(sq: str):
    c, r = sq[0], sq[1]
    return (8 - int(r), COLS.index(c))

def sq(row, col):
    return COLS[col] + str(8 - row)

def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8

# ── Fast attack detection ─────────────────────────────────────────────────────
_KNIGHT_OFFSETS = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
_KING_OFFSETS   = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_DIAG_DIRS      = [(-1,-1),(-1,1),(1,-1),(1,1)]
_CROSS_DIRS     = [(-1,0),(1,0),(0,-1),(0,1)]

def square_attacked_by(board, tr, tc, color):
    """Return True if (tr,tc) is attacked by any piece of `color`. Fast path."""
    opp = 'b' if color == 'w' else 'w'
    g   = board.grid
    # Knights
    for dr, dc in _KNIGHT_OFFSETS:
        nr, nc = tr+dr, tc+dc
        if 0<=nr<8 and 0<=nc<8:
            p = g[nr][nc]
            if p and p.color == color and p.kind == 'N':
                return True
    # Kings
    for dr, dc in _KING_OFFSETS:
        nr, nc = tr+dr, tc+dc
        if 0<=nr<8 and 0<=nc<8:
            p = g[nr][nc]
            if p and p.color == color and p.kind == 'K':
                return True
    # Diagonals (B/Q)
    for dr, dc in _DIAG_DIRS:
        nr, nc = tr+dr, tc+dc
        while 0<=nr<8 and 0<=nc<8:
            p = g[nr][nc]
            if p:
                if p.color == color and p.kind in ('B','Q'):
                    return True
                break
            nr+=dr; nc+=dc
    # Ranks/files (R/Q)
    for dr, dc in _CROSS_DIRS:
        nr, nc = tr+dr, tc+dc
        while 0<=nr<8 and 0<=nc<8:
            p = g[nr][nc]
            if p:
                if p.color == color and p.kind in ('R','Q'):
                    return True
                break
            nr+=dr; nc+=dc
    # Pawns
    pd = 1 if color == 'w' else -1   # direction FROM which pawn attacks
    for dc in (-1, 1):
        nr, nc = tr+pd, tc+dc
        if 0<=nr<8 and 0<=nc<8:
            p = g[nr][nc]
            if p and p.color == color and p.kind == 'P':
                return True
    return False


# ── Piece ─────────────────────────────────────────────────────────────────────
class Piece:
    __slots__ = ('kind', 'color', 'moved')
    def __init__(self, kind, color):
        self.kind  = kind
        self.color = color
        self.moved = False

    def __repr__(self):
        return UNICODE[(self.kind, self.color)]

    def copy(self):
        p = Piece(self.kind, self.color)
        p.moved = self.moved
        return p

# ── Board ─────────────────────────────────────────────────────────────────────
class Board:
    def __init__(self):
        self.grid = [[None] * 8 for _ in range(8)]
        self.en_passant = None
        self.half_moves = 0
        self.full_moves = 1
        self._setup()

    def _setup(self):
        back = ['R','N','B','Q','K','B','N','R']
        for col, kind in enumerate(back):
            self.grid[0][col] = Piece(kind, 'b')
            self.grid[7][col] = Piece(kind, 'w')
        for col in range(8):
            self.grid[1][col] = Piece('P', 'b')
            self.grid[6][col] = Piece('P', 'w')

    def get(self, r, c):   return self.grid[r][c]
    def set(self, r, c, p): self.grid[r][c] = p

    def find_king(self, color):
        for r in range(8):
            for c in range(8):
                p = self.grid[r][c]
                if p and p.kind == 'K' and p.color == color:
                    return (r, c)
        return None

    def copy(self):
        b = Board.__new__(Board)
        b.grid = [[p.copy() if p else None for p in row] for row in self.grid]
        b.en_passant = self.en_passant
        b.half_moves = self.half_moves
        b.full_moves = self.full_moves
        return b

    def material(self):
        """Return (white_score, black_score) using standard chess point values."""
        SCORES = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        w = b = 0
        for row in self.grid:
            for p in row:
                if p:
                    if p.color == 'w': w += SCORES[p.kind]
                    else:              b += SCORES[p.kind]
        return w, b

    def score_bar(self, perspective='w'):
        """One-line score display for each side: points + captured pieces + advantage."""
        SCORES = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        ICONS  = {'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛'}
        RESET  = '\033[0m'
        BOLD   = '\033[1m'
        DIM    = '\033[2m'
        GREEN  = '\033[32m'

        START    = {'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1}
        on_board = {'w': {k: 0 for k in START}, 'b': {k: 0 for k in START}}
        for row in self.grid:
            for p in row:
                if p and p.kind in START:
                    on_board[p.color][p.kind] += 1

        # captured[c] = pieces color c has captured (missing from opponent)
        captured = {
            'w': {k: START[k] - on_board['b'][k] for k in START},
            'b': {k: START[k] - on_board['w'][k] for k in START},
        }

        w_score, b_score = self.material()
        diff = w_score - b_score

        def bar(color):
            caps       = captured[color]
            parts      = [ICONS[k] * caps[k] for k in ('Q','R','B','N','P') if caps[k] > 0]
            pieces_str = ' '.join(parts) if parts else ''
            adv        = diff if color == 'w' else -diff
            adv_str    = f' {GREEN}{BOLD}+{adv}{RESET}' if adv > 0 else ''
            label      = 'White' if color == 'w' else 'Black'
            line       = f"  {BOLD}{label}{RESET}"
            if pieces_str or adv_str:
                line  += f"  {pieces_str}{adv_str}"
            return line

        top = 'b' if perspective == 'w' else 'w'
        bot = 'w' if perspective == 'w' else 'b'
        return bar(top), bar(bot)

    def display(self, perspective='w'):
        rows = range(8) if perspective == 'w' else range(7, -1, -1)
        cols = range(8) if perspective == 'w' else range(7, -1, -1)
        LIGHT = '\033[48;5;223m'
        DARK  = '\033[48;5;136m'
        RESET = '\033[0m'
        BOLD  = '\033[1m'
        print()
        top_bar, bot_bar = self.score_bar(perspective)
        print(top_bar)
        col_labels = ' ' + '  '.join(COLS[c] for c in cols)
        print(f"  {BOLD}{col_labels}{RESET}")
        for r in rows:
            rank = 8 - r
            row_str = f"{BOLD}{rank}{RESET} "
            for c in cols:
                light = (r + c) % 2 == 0
                bg = LIGHT if light else DARK
                piece = self.grid[r][c]
                symbol = str(piece) if piece else ' '
                row_str += f"{bg} {symbol} {RESET}"
            row_str += f" {BOLD}{rank}{RESET}"
            print(row_str)
        print(f"  {BOLD}{col_labels}{RESET}")
        print(bot_bar)

# ── Move generation ───────────────────────────────────────────────────────────
def raw_moves(board, r, c):
    piece = board.get(r, c)
    if piece is None: return []
    kind, color = piece.kind, piece.color
    opp = 'b' if color == 'w' else 'w'
    moves = []

    def add(dr, dc, slide=False):
        nr, nc = r + dr, c + dc
        while in_bounds(nr, nc):
            target = board.get(nr, nc)
            if target is None:
                moves.append((nr, nc, None))
            elif target.color == opp:
                moves.append((nr, nc, None))
                break
            else:
                break
            if not slide: break
            nr += dr; nc += dc

    if kind == 'P':
        direction = -1 if color == 'w' else 1
        nr = r + direction
        if in_bounds(nr, c) and board.get(nr, c) is None:
            moves.append((nr, c, None))
            start_rank = 6 if color == 'w' else 1
            nr2 = r + 2 * direction
            if r == start_rank and board.get(nr2, c) is None:
                moves.append((nr2, c, None))
        for dc in (-1, 1):
            nc = c + dc
            if in_bounds(nr, nc):
                target = board.get(nr, nc)
                if target and target.color == opp:
                    moves.append((nr, nc, None))
                if board.en_passant == sq(nr, nc):
                    moves.append((nr, nc, 'ep'))
    elif kind == 'N':
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            add(dr, dc)
    elif kind == 'B':
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]: add(dr, dc, True)
    elif kind == 'R':
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: add(dr, dc, True)
    elif kind == 'Q':
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]: add(dr, dc, True)
    elif kind == 'K':
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]: add(dr, dc)
        if not piece.moved:
            back_rank = 7 if color == 'w' else 0
            if r == back_rank:
                rook = board.get(back_rank, 7)
                if (rook and rook.kind == 'R' and not rook.moved and
                        board.get(back_rank, 5) is None and board.get(back_rank, 6) is None):
                    moves.append((back_rank, 6, 'castle_k'))
                rook = board.get(back_rank, 0)
                if (rook and rook.kind == 'R' and not rook.moved and
                        board.get(back_rank, 1) is None and
                        board.get(back_rank, 2) is None and board.get(back_rank, 3) is None):
                    moves.append((back_rank, 2, 'castle_q'))
    return moves


def is_in_check(board, color):
    pos = board.find_king(color)
    if pos is None: return False
    kr, kc = pos
    opp = 'b' if color == 'w' else 'w'
    return square_attacked_by(board, kr, kc, opp)


def apply_move(board, from_r, from_c, to_r, to_c, flag, promotion='Q'):
    b = board.copy()
    piece = b.get(from_r, from_c)
    b.en_passant = None

    if flag == 'ep':
        direction = -1 if piece.color == 'w' else 1
        b.set(to_r - direction, to_c, None)
    elif flag == 'castle_k':
        rook = b.get(to_r, 7)
        b.set(to_r, 7, None); b.set(to_r, 5, rook); rook.moved = True
        for col in (5, 6):   # f1/f8 and g1/g8 — don't include starting square
            tb = b.copy(); tb.set(to_r, col, piece); tb.set(from_r, from_c, None)
            if is_in_check(tb, piece.color): return None
    elif flag == 'castle_q':
        rook = b.get(to_r, 0)
        b.set(to_r, 0, None); b.set(to_r, 3, rook); rook.moved = True
        for col in (3, 2):   # d1/d8 and c1/c8 — don't include starting square
            tb = b.copy(); tb.set(to_r, col, piece); tb.set(from_r, from_c, None)
            if is_in_check(tb, piece.color): return None

    if piece.kind == 'P' and abs(to_r - from_r) == 2:
        mid = (from_r + to_r) // 2
        b.en_passant = sq(mid, to_c)

    b.set(to_r, to_c, piece); b.set(from_r, from_c, None)
    piece.moved = True

    if piece.kind == 'P' and (to_r == 0 or to_r == 7):
        b.set(to_r, to_c, Piece(promotion.upper(), piece.color))

    if is_in_check(b, piece.color): return None
    return b


def legal_moves(board, color):
    result = []
    for r in range(8):
        for c in range(8):
            p = board.get(r, c)
            if p and p.color == color:
                for to_r, to_c, flag in raw_moves(board, r, c):
                    nb = apply_move(board, r, c, to_r, to_c, flag)
                    if nb is not None:
                        result.append((r, c, to_r, to_c, flag))
    return result

# ── Zobrist hashing for transposition table ───────────────────────────────────
import random as _random
_rng = _random.Random(20250527)
_ZOBRIST = {}
for _r in range(8):
    for _c in range(8):
        for _kind in 'KQRBNP':
            for _col in ('w', 'b'):
                _ZOBRIST[(_r, _c, _kind, _col)] = _rng.getrandbits(64)
_ZOBRIST['w_turn'] = _rng.getrandbits(64)

def zobrist_hash(board, color):
    h = 0
    for r in range(8):
        for c in range(8):
            p = board.get(r, c)
            if p:
                h ^= _ZOBRIST[(r, c, p.kind, p.color)]
    if color == 'w':
        h ^= _ZOBRIST['w_turn']
    return h

# ── Transposition table ───────────────────────────────────────────────────────
# Entry: (depth, flag, value, best_move)   flag: 'exact' | 'lower' | 'upper'
_TT = {}
TT_SIZE = 1 << 20   # ~1M entries

def tt_store(key, depth, flag, value, move):
    _TT[key % TT_SIZE] = (key, depth, flag, value, move)

def tt_probe(key, depth, alpha, beta):
    entry = _TT.get(key % TT_SIZE)
    if entry is None or entry[0] != key:   # strict full-key collision check
        return None, None
    _, stored_depth, flag, value, move = entry
    if stored_depth >= depth:              # only trust deep enough entries
        if flag == 'exact':                    return value, move
        if flag == 'lower' and value >= beta:  return value, move
        if flag == 'upper' and value <= alpha: return value, move
    return None, move                      # move hint only, no score


# ══════════════════════════════════════════════════════════════════════════════
# ── Neural Network (AlphaZero-style self-learning evaluator) ──────────────────
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chess_weights.npz')
REPLAY_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chess_replay.pkl')

# ── Board encoding ───────────────────────────────────────────────────────────
_NN_INPUT  = 776    # 12 planes × 64 squares + 8 features
_PIECE_PLANE = {
    ('K','w'):0,('Q','w'):1,('R','w'):2,('B','w'):3,('N','w'):4,('P','w'):5,
    ('K','b'):6,('Q','b'):7,('R','b'):8,('B','b'):9,('N','b'):10,('P','b'):11,
}
_PVAL_NN = {'P':1,'N':3,'B':3,'R':5,'Q':9,'K':0}

def encode_board(board, color):
    """Encode board to 776-d float32 vector."""
    vec = np.zeros(_NN_INPUT, dtype=np.float32)
    w_mat = b_mat = 0
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p:
                plane = _PIECE_PLANE.get((p.kind, p.color))
                if plane is not None:
                    vec[plane * 64 + r * 8 + c] = 1.0
                if p.color == 'w': w_mat += _PVAL_NN[p.kind]
                else:              b_mat += _PVAL_NN[p.kind]
    vec[768] = 1.0 if color == 'w' else -1.0
    vec[769] = (w_mat - b_mat) / 39.0
    vec[770] = w_mat / 39.0
    vec[771] = b_mat / 39.0
    vec[772] = min(board.full_moves / 100.0, 1.0)
    wk = board.find_king('w'); bk = board.find_king('b')
    vec[773] = 1.0 if (wk and wk[0] == 7) else 0.0
    vec[774] = 1.0 if (bk and bk[0] == 0) else 0.0
    vec[775] = 1.0 if board.en_passant else 0.0
    return vec


# ── PyTorch network (GPU) ─────────────────────────────────────────────────────
if _TORCH_OK:
    # Detect GPU — prefers CUDA (Nvidia), then MPS (Apple), then CPU
    if torch.cuda.is_available():
        _DEVICE = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _DEVICE = torch.device('mps')
    else:
        _DEVICE = torch.device('cpu')

    class _TorchNet(nn.Module):
        """
        Deep residual network for board evaluation.
        Bigger than the old numpy net — benefits from GPU.
          Input  : 776
          Hidden : 512 → 512 → 256 → 128
          Output : 1  (tanh → -1..+1)
        """
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(776, 512), nn.LayerNorm(512), nn.GELU(),
                nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Linear(256, 128), nn.GELU(),
                nn.Linear(128,   1), nn.Tanh(),
            )
            # He initialisation
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x)


class ChessNet:
    """
    Chess evaluation network.
    Uses PyTorch + GPU when available, falls back to numpy on CPU.
    Trained via TD(λ) self-play + experience replay.
    """
    def __init__(self):
        self.games_trained = 0
        self.lam           = 0.7

        if _TORCH_OK:
            self._torch = _TorchNet().to(_DEVICE)
            self._opt   = optim.Adam(self._torch.parameters(), lr=1e-3,
                                     weight_decay=1e-4)
            self._sched = optim.lr_scheduler.StepLR(self._opt,
                                                     step_size=500, gamma=0.5)
            self._use_torch = True
            # TD trace tensors (one per parameter)
            self._traces = [torch.zeros_like(p) for p in self._torch.parameters()]
            self._prev_v = None
        else:
            # Numpy fallback (same as before)
            self._use_torch = False
            self._build_numpy()
            self._init_velocity()

    # ── Numpy fallback ────────────────────────────────────────────────────
    def _build_numpy(self):
        s1=np.sqrt(2.0/776); s2=np.sqrt(2.0/256); s3=np.sqrt(2.0/128)
        self.W1=np.random.randn(776,256)*s1; self.b1=np.zeros(256)
        self.W2=np.random.randn(256,128)*s2; self.b2=np.zeros(128)
        self.W3=np.random.randn(128,  1)*s3; self.b3=np.zeros(1)

    def _init_velocity(self):
        self.vW1=np.zeros_like(self.W1); self.vb1=np.zeros_like(self.b1)
        self.vW2=np.zeros_like(self.W2); self.vb2=np.zeros_like(self.b2)
        self.vW3=np.zeros_like(self.W3); self.vb3=np.zeros_like(self.b3)

    def _np_forward(self, x):
        self._x  = x
        self._h1 = np.maximum(0, x @ self.W1 + self.b1)
        self._h2 = np.maximum(0, self._h1 @ self.W2 + self.b2)
        out = np.tanh(self._h2 @ self.W3 + self.b3)
        self._out = out; return float(out[0])

    def _np_backward(self, delta):
        d3=delta*(1-self._out**2); gW3=np.outer(self._h2,d3); gb3=d3
        d2=(d3@self.W3.T)*(self._h2>0); gW2=np.outer(self._h1,d2); gb2=d2
        d1=(d2@self.W2.T)*(self._h1>0); gW1=np.outer(self._x,d1);  gb1=d1
        return gW1,gb1,gW2,gb2,gW3,gb3

    def _np_update(self, gW1,gb1,gW2,gb2,gW3,gb3, lr=0.001):
        m=0.9
        self.vW1=m*self.vW1+lr*gW1; self.W1+=self.vW1
        self.vb1=m*self.vb1+lr*gb1; self.b1+=self.vb1
        self.vW2=m*self.vW2+lr*gW2; self.W2+=self.vW2
        self.vb2=m*self.vb2+lr*gb2; self.b2+=self.vb2
        self.vW3=m*self.vW3+lr*gW3; self.W3+=self.vW3
        self.vb3=m*self.vb3+lr*gb3; self.b3+=self.vb3

    # ── Public API ────────────────────────────────────────────────────────
    def predict(self, board, color):
        """Return centipawn score (+ve = white better)."""
        enc = encode_board(board, color)
        if self._use_torch:
            with torch.no_grad():
                x   = torch.tensor(enc, dtype=torch.float32).to(_DEVICE)
                val = self._torch(x.unsqueeze(0)).item()
        else:
            val = self._np_forward(enc)
        return int(val * 900)

    # ── TD(λ) step (called during self-play) ──────────────────────────────
    def td_step(self, board, color, reward=None):
        enc = encode_board(board, color)
        if self._use_torch:
            x   = torch.tensor(enc, dtype=torch.float32).to(_DEVICE)
            out = self._torch(x.unsqueeze(0))
            v   = out.item()
            if self._prev_v is None:
                self._prev_v = v
                # zero traces
                for t in self._traces: t.zero_()
                return
            target = reward if reward is not None else v
            delta  = torch.tensor(target - self._prev_v, dtype=torch.float32)
            # Compute gradients w.r.t. output
            self._opt.zero_grad()
            out.backward(torch.ones(1,1).to(_DEVICE))
            # Update eligibility traces and weights
            with torch.no_grad():
                for p, tr in zip(self._torch.parameters(), self._traces):
                    if p.grad is not None:
                        tr.mul_(self.lam).add_(p.grad)
                        p.add_(1e-3 * delta * tr)
                        p.grad.zero_()
            self._prev_v = v
        else:
            # numpy TD
            v = self._np_forward(enc)
            if not hasattr(self, '_np_traces') or self._np_traces is None:
                self._np_traces = [np.zeros_like(w) for w in
                                   (self.W1,self.b1,self.W2,self.b2,self.W3,self.b3)]
                self._prev_v = v; return
            target = reward if reward is not None else v
            delta  = target - self._prev_v
            grads  = self._np_backward(1.0)
            for i,g in enumerate(grads):
                self._np_traces[i] = self.lam * self._np_traces[i] + g
            self._np_update(*[delta*t for t in self._np_traces])
            self._prev_v = v
            self._np_forward(enc)

    def td_reset(self):
        self._prev_v = None
        if self._use_torch:
            for t in self._traces: t.zero_()
        else:
            self._np_traces = None

    def td_finish(self, board, outcome):
        self.td_step(board, 'w', reward=float(outcome))
        self.games_trained += 1
        self.save()
        self.td_reset()

    # ── Batch training from replay buffer ─────────────────────────────────
    def batch_train(self, samples):
        """Train on a list of (encoding, outcome) pairs using GPU."""
        if not samples:
            return
        if self._use_torch:
            encs    = np.stack([s[0] for s in samples])
            targets = np.array([s[1] for s in samples], dtype=np.float32)
            x = torch.tensor(encs,    dtype=torch.float32).to(_DEVICE)
            y = torch.tensor(targets, dtype=torch.float32).to(_DEVICE).unsqueeze(1)
            # Multiple gradient steps per batch
            for _ in range(4):
                self._opt.zero_grad()
                pred = self._torch(x)
                loss = nn.functional.mse_loss(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self._torch.parameters(), 1.0)
                self._opt.step()
            self._sched.step()
        else:
            for enc, outcome in samples:
                pred  = self._np_forward(enc)
                delta = outcome - pred
                self._np_update(*self._np_backward(delta), lr=0.0005)

    # ── Save / Load ───────────────────────────────────────────────────────
    def save(self):
        if self._use_torch:
            torch.save({
                'model': self._torch.state_dict(),
                'opt':   self._opt.state_dict(),
                'games': self.games_trained,
            }, WEIGHTS_FILE.replace('.npz', '.pt'))
        else:
            np.savez(WEIGHTS_FILE,
                     W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                     W3=self.W3, b3=self.b3,
                     games=np.array([self.games_trained]))

    def load(self):
        pt_file  = WEIGHTS_FILE.replace('.npz', '.pt')
        npz_file = WEIGHTS_FILE
        if self._use_torch and os.path.exists(pt_file):
            try:
                ck = torch.load(pt_file, map_location=_DEVICE)
                self._torch.load_state_dict(ck['model'])
                self._opt.load_state_dict(ck['opt'])
                self.games_trained = ck.get('games', 0)
                return True
            except Exception:
                pass
        if os.path.exists(npz_file):
            try:
                d = np.load(npz_file)
                if not self._use_torch:
                    self.W1=d['W1']; self.b1=d['b1']
                    self.W2=d['W2']; self.b2=d['b2']
                    self.W3=d['W3']; self.b3=d['b3']
                self.games_trained = int(d['games'][0]) if 'games' in d else 0
                return True
            except Exception:
                pass
        return False


class TDTrainer:
    """Thin wrapper kept for compatibility — delegates to ChessNet."""
    def __init__(self, net): self.net = net
    def reset(self):          self.net.td_reset()
    def step(self, b, c, reward=None): self.net.td_step(b, c, reward)
    def finish(self, b, outcome):      self.net.td_finish(b, outcome)


class ReplayBuffer:
    """Experience replay buffer — stores (encoding, outcome) pairs."""
    def __init__(self, maxlen=100000):   # bigger buffer for GPU batch training
        self.buf    = []
        self.maxlen = maxlen

    def push(self, enc, outcome):
        if len(self.buf) >= self.maxlen:
            self.buf.pop(0)
        self.buf.append((enc, float(outcome)))

    def sample(self, n=1024):           # bigger batches for GPU
        idx = np.random.choice(len(self.buf), min(n, len(self.buf)), replace=False)
        return [self.buf[i] for i in idx]

    def save(self):
        try:
            with open(REPLAY_FILE, 'wb') as f:
                pickle.dump(self.buf, f)
        except Exception:
            pass

    def load(self):
        if os.path.exists(REPLAY_FILE):
            try:
                with open(REPLAY_FILE, 'rb') as f:
                    self.buf = pickle.load(f)
                return True
            except Exception:
                pass
        return False


def _batch_train(net, replay, **kwargs):
    """Pull a batch from replay and train."""
    if len(replay.buf) >= 256:
        net.batch_train(replay.sample(1024 if net._use_torch else 256))


# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_CHESS_NET  = None
_REPLAY_BUF = None

def _get_net():
    global _CHESS_NET
    if _CHESS_NET is None:
        _CHESS_NET = ChessNet()
        _CHESS_NET.load()
    return _CHESS_NET

def _get_replay():
    global _REPLAY_BUF
    if _REPLAY_BUF is None:
        _REPLAY_BUF = ReplayBuffer()
        _REPLAY_BUF.load()
    return _REPLAY_BUF

def nn_games_trained():
    return _get_net().games_trained

# ── Static evaluation ─────────────────────────────────────────────────────────
def is_endgame(board):
    queens = minor = 0
    for r in range(8):
        for c in range(8):
            p = board.get(r, c)
            if p:
                if p.kind == 'Q': queens += 1
                elif p.kind in ('N','B','R'): minor += 1
    return queens == 0 or (queens <= 2 and minor <= 2)


def build_attack_map(board, color):
    """
    Returns a dict {(r,c): count} of squares attacked by `color`.
    Also returns list of (attacker_piece, to_r, to_c) for tactical use.
    Uses pre-computed raw_moves — called once per evaluate().
    """
    attacks = {}
    attacker_list = []
    g = board.grid
    for r in range(8):
        for c in range(8):
            p = g[r][c]
            if p and p.color == color:
                for mr, mc, _ in raw_moves(board, r, c):
                    key = (mr, mc)
                    if key in attacks: attacks[key] += 1
                    else:              attacks[key]  = 1
                    attacker_list.append((p, mr, mc))
    return attacks, attacker_list


def tactical_score(board, color, atk_map, atk_list, opp_atk_map):
    """
    Reward forks, penalize hanging pieces, bonus for king attacks.
    atk_list = [(piece, to_r, to_c)] for our pieces.
    """
    opp   = 'b' if color == 'w' else 'w'
    score = 0
    g     = board.grid

    # ── Hanging piece penalty ─────────────────────────────────────────────
    for r in range(8):
        for c in range(8):
            p = g[r][c]
            if p and p.color == color:
                atkers = opp_atk_map.get((r, c), 0)
                defnds = atk_map.get((r, c), 0)
                if atkers > 0 and defnds == 0:
                    score -= PIECE_VALUE[p.kind] // 2   # undefended
                elif atkers > defnds and p.kind != 'K':
                    score -= PIECE_VALUE[p.kind] // 5   # outgunned

    # ── Fork detection ────────────────────────────────────────────────────
    # Group attacked enemy squares by the piece doing the attacking
    # Key = (from_r, from_c) of attacker
    from_sq = {}
    for p, mr, mc in atk_list:
        target = g[mr][mc]
        if target and target.color == opp and PIECE_VALUE[target.kind] >= 300:
            # Find attacker location (piece object identity trick)
            for rr in range(8):
                for cc in range(8):
                    if g[rr][cc] is p:
                        key = (rr, cc)
                        if key not in from_sq: from_sq[key] = []
                        from_sq[key].append(PIECE_VALUE[target.kind])
                        break
                else: continue
                break

    for (fr, fc), tvals in from_sq.items():
        if len(tvals) >= 2:
            tvals.sort(reverse=True)
            score += tvals[1] // 3   # bonus = 2nd most valuable target / 3

    # ── King zone pressure ────────────────────────────────────────────────
    opp_kr, opp_kc = board.find_king(opp)
    if opp_kr is not None:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = opp_kr + dr, opp_kc + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    score += atk_map.get((nr, nc), 0) * 5

    return score


def king_safety(board, color, endgame, opp_atk_map):
    """Penalty for exposed king; bonus for pawn shield."""
    if endgame:
        return 0
    kr, kc = board.find_king(color)
    if kr is None:
        return 0
    score    = 0
    opp      = 'b' if color == 'w' else 'w'
    direction = -1 if color == 'w' else 1

    # Pawn shield
    for dc in (-1, 0, 1):
        nc, nr = kc + dc, kr + direction
        if in_bounds(nr, nc):
            p = board.get(nr, nc)
            if p and p.kind == 'P' and p.color == color:
                score += 15
            elif p and p.kind == 'P' and p.color == opp:
                score -= 20
    # Open files near king
    for dc in (-1, 0, 1):
        nc = kc + dc
        if 0 <= nc < 8:
            own_pawn = any(
                (pp := board.get(rr, nc)) and pp.kind == 'P' and pp.color == color
                for rr in range(8))
            if not own_pawn:
                score -= 25
    # Enemy pieces swarming king zone
    king_zone_attacks = sum(
        opp_atk_map.get((kr + dr, kc + dc), 0)
        for dr in range(-2, 3) for dc in range(-2, 3)
        if in_bounds(kr + dr, kc + dc)
    )
    score -= king_zone_attacks * 8
    # King in centre (middlegame)
    if 2 <= kc <= 5 and 2 <= kr <= 5:
        score -= 50
    return score


def evaluate(board):
    """
    Positive = good for White, negative = good for Black.
    Material + PST + pawn structure + king safety + mobility +
    bishop pair + rook files + forks + hanging pieces.
    """
    # Endgame detection inline (avoid second board scan)
    queens = minor = 0
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p:
                if p.kind == 'Q': queens += 1
                elif p.kind in ('N','B','R'): minor += 1
    endgame   = queens == 0 or (queens <= 2 and minor <= 2)
    score     = 0
    w_bishops = b_bishops = 0
    w_mob = b_mob = 0

    # Build attack maps once for the whole evaluation
    w_atk, w_atk_list = build_attack_map(board, 'w')
    b_atk, b_atk_list = build_attack_map(board, 'b')

    for r in range(8):
        for c in range(8):
            p = board.get(r, c)
            if p is None:
                continue
            sign = 1 if p.color == 'w' else -1
            val  = PIECE_VALUE[p.kind]
            val += pst_score(p.kind, p.color, r, c, endgame)
            # Mobility counted here (pseudo-legal, fast)
            mob = len(raw_moves(board, r, c))
            if p.color == 'w': w_mob += mob
            else:              b_mob += mob

            if p.kind == 'P':
                opp_c = 'b' if p.color == 'w' else 'w'
                dirn  = -1 if p.color == 'w' else 1
                # Doubled pawns
                col_pawns = sum(1 for rr in range(8)
                    if (pp := board.get(rr, c)) and pp.kind == 'P' and pp.color == p.color)
                if col_pawns > 1: val -= 25
                # Isolated pawns
                neighbour = any(
                    any((pp := board.get(rr, c + dc)) and pp.kind == 'P' and pp.color == p.color
                        for rr in range(8))
                    for dc in (-1, 1) if 0 <= c + dc < 8
                )
                if not neighbour: val -= 20
                # Passed pawn
                passed = not any(
                    (pp := board.get(rr, c + dc)) and pp.kind == 'P' and pp.color == opp_c
                    for rr in range(r + dirn, (0 if dirn == -1 else 8), dirn)
                    for dc in (-1, 0, 1) if 0 <= c + dc < 8
                )
                if passed:
                    advance = (6 - r) if p.color == 'w' else (r - 1)
                    val += 20 + advance * 15
                    # Extra bonus if supported by own pieces
                    own_atk = w_atk if p.color == 'w' else b_atk
                    if own_atk.get((r, c), 0) > 0:
                        val += 10

            elif p.kind == 'B':
                if p.color == 'w': w_bishops += 1
                else:              b_bishops += 1
                # Penalty for pawns on same colour
                sq_color = (r + c) % 2
                same_col_pawns = sum(
                    1 for rr in range(8) for cc in range(8)
                    if (rr + cc) % 2 == sq_color
                    and (pp := board.get(rr, cc)) and pp.kind == 'P' and pp.color == p.color
                )
                val -= same_col_pawns * 3

            elif p.kind == 'R':
                own_p = any((pp := board.get(rr, c)) and pp.kind=='P' and pp.color==p.color for rr in range(8))
                opp_p = any((pp := board.get(rr, c)) and pp.kind=='P' and pp.color!=p.color for rr in range(8))
                if not own_p and not opp_p: val += 30
                elif not own_p:             val += 15
                # Rook on 7th rank
                seventh = 1 if p.color == 'w' else 6
                if r == seventh: val += 20

            elif p.kind == 'N':
                opp_c = 'b' if p.color == 'w' else 'w'
                dirn  = 1 if p.color == 'w' else -1
                # Outpost
                centre = {(2,2),(2,3),(2,4),(2,5),(3,2),(3,3),(3,4),(3,5),
                          (4,2),(4,3),(4,4),(4,5),(5,2),(5,3),(5,4),(5,5)}
                if (r, c) in centre:
                    attackable = any(
                        (pp := board.get(r+dirn, c+dc)) and pp.kind=='P' and pp.color==opp_c
                        for dc in (-1, 1) if in_bounds(r+dirn, c+dc)
                    )
                    if not attackable: val += 20

            score += sign * val

    if w_bishops >= 2: score += 40
    if b_bishops >= 2: score -= 40

    # Mobility already counted inline above
    score += (w_mob - b_mob) * 2

    # King safety (uses attack maps)
    score += king_safety(board, 'w', endgame, b_atk)
    score -= king_safety(board, 'b', endgame, w_atk)

    # Tactical: forks and hanging pieces
    score += tactical_score(board, 'w', w_atk, w_atk_list, b_atk)
    score -= tactical_score(board, 'b', b_atk, b_atk_list, w_atk)

    # ── Neural network blend (if trained) ────────────────────────────────
    net = _get_net()
    if net.games_trained >= 20:
        # Blend: more weight to NN as it accumulates games (caps at 60%)
        nn_weight = min(0.6, net.games_trained / 500.0)
        nn_score  = net.predict(board, 'w')
        score     = int(score * (1 - nn_weight) + nn_score * nn_weight)

    return score


# ── Static Exchange Evaluation (SEE) ─────────────────────────────────────────
def _least_attacker(board, tr, tc, color):
    """Find least-valuable piece of `color` that attacks (tr,tc). Returns (r,c,val) or None."""
    best = None
    for r in range(8):
        for c in range(8):
            p = board.get(r, c)
            if p and p.color == color:
                for mr, mc, _ in raw_moves(board, r, c):
                    if (mr, mc) == (tr, tc):
                        v = PIECE_VALUE[p.kind]
                        if best is None or v < best[2]:
                            best = (r, c, v)
    return best


def see(board, to_r, to_c, target_val, from_r, from_c, attacker_val):
    """
    Static Exchange Evaluation.
    Returns material gain (positive = winning, negative = losing).
    Works on a lightweight square-copy to avoid full Board.copy() overhead.
    """
    # Work on a simple grid copy (list of lists of pieces)
    grid = [row[:] for row in board.grid]

    gain  = [0] * 32
    depth = 0
    gain[0] = target_val

    # First capture: move from_r,from_c -> to_r,to_c
    attacker      = grid[from_r][from_c]
    grid[to_r][to_c] = attacker
    grid[from_r][from_c] = None
    color = attacker.color
    depth = 1

    class _B:
        """Thin wrapper so raw_moves/find_king work on the grid."""
        def __init__(self, g): self.grid = g; self.en_passant = None
        def get(self, r, c): return self.grid[r][c]
        def set(self, r, c, p): self.grid[r][c] = p

    b = _B(grid)

    while True:
        color = 'b' if color == 'w' else 'w'
        result = _least_attacker(b, to_r, to_c, color)
        if result is None:
            break
        ar, ac, aval = result
        gain[depth] = aval - gain[depth - 1]
        # Stand-pat: if taking is worse, stop
        if max(-gain[depth - 1], gain[depth]) < 0:
            break
        att = b.get(ar, ac)
        b.set(to_r, to_c, att)
        b.set(ar, ac, None)
        depth += 1
        if depth >= 32:
            break

    while depth > 1:
        depth -= 1
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])

    return gain[0]


def see_capture(board, move):
    """Return SEE value for a capture move. 0 for non-captures."""
    r0, c0, r1, c1, flag = move
    victim = board.get(r1, c1)
    if victim is None and flag != 'ep':
        return 0
    target_val  = 100 if flag == 'ep' else PIECE_VALUE[victim.kind]
    attacker    = board.get(r0, c0)
    return see(board, r1, c1, target_val, r0, c0, PIECE_VALUE[attacker.kind])


def is_losing_capture(board, move):
    return see_capture(board, move) < 0


def piece_is_hanging(board, move):
    """
    Fast: does the piece land where a CHEAPER enemy can take it?
    Uses pre-move board — no copy needed.
    """
    r0, c0, r1, c1, flag = move
    piece = board.get(r0, c0)
    if piece is None or flag in ('castle_k', 'castle_q'):
        return False
    opp = 'b' if piece.color == 'w' else 'w'
    for r in range(8):
        for c in range(8):
            if (r, c) == (r0, c0):
                continue
            p = board.get(r, c)
            if p and p.color == opp and PIECE_VALUE[p.kind] <= PIECE_VALUE[piece.kind]:
                for mr, mc, _ in raw_moves(board, r, c):
                    if (mr, mc) == (r1, c1):
                        return True
    return False

# ── Move ordering ─────────────────────────────────────────────────────────────
# Killer move table: killers[ply] = [move1, move2]
_KILLERS  = [[None, None] for _ in range(64)]
# History heuristic table: history[color][from_r][from_c][to_r][to_c]
_HISTORY  = {'w': [[[[0]*8 for _ in range(8)] for _ in range(8)] for _ in range(8)],
             'b': [[[[0]*8 for _ in range(8)] for _ in range(8)] for _ in range(8)]}

def update_killer(ply, move):
    if _KILLERS[ply][0] != move:
        _KILLERS[ply][1] = _KILLERS[ply][0]
        _KILLERS[ply][0] = move

def move_score(board, move, ply=0, tt_move=None, color='w'):
    r0, c0, r1, c1, flag = move
    attacker = board.get(r0, c0)
    victim   = board.get(r1, c1)
    score    = 0

    # TT move first — always search this first
    if move == tt_move:
        return 1_000_000

    # Captures: winning/equal captures first, then quiet, losing captures last
    if victim or flag == 'ep':
        target_val = 100 if flag == 'ep' else PIECE_VALUE[victim.kind]
        see_val    = see(board, r1, c1, target_val, r0, c0, PIECE_VALUE[attacker.kind])
        if see_val > 0:
            score += 100_000 + see_val        # winning capture
        elif see_val == 0:
            score += 50_000                   # equal exchange
        else:
            score += 10_000 + see_val         # losing capture — still above quiet but ordered by loss size

    # promotions
    if attacker.kind == 'P' and (r1 == 0 or r1 == 7):
        score += 90_000

    # castling
    if flag in ('castle_k', 'castle_q'):
        score += 5_000

    # killer moves (quiet moves that caused beta cutoffs)
    if ply < 64:
        if move == _KILLERS[ply][0]: score += 9_000
        elif move == _KILLERS[ply][1]: score += 8_000

    # history heuristic for quiet moves
    if not victim and flag not in ('ep',):
        score += min(_HISTORY[color][r0][c0][r1][c1], 5_000)

    # centre
    if (r1, c1) in ((3,3),(3,4),(4,3),(4,4)):
        score += 50

    return score


def order_moves(board, moves, ply=0, tt_move=None, color='w'):
    return sorted(moves,
                  key=lambda m: move_score(board, m, ply, tt_move, color),
                  reverse=True)

# ── Quiescence search ─────────────────────────────────────────────────────────
def quiescence(board, alpha, beta, color, depth=0):
    opp = 'b' if color == 'w' else 'w'

    in_check  = is_in_check(board, color)
    stand_pat = evaluate(board) if color == 'w' else -evaluate(board)

    if not in_check:
        if stand_pat >= beta: return beta
        # Delta pruning: if even capturing the best piece won't raise alpha, bail
        DELTA = 900  # queen value
        if stand_pat + DELTA < alpha: return alpha
        if stand_pat > alpha: alpha = stand_pat

    if depth > 6: return alpha

    moves = legal_moves(board, color)
    if not moves:
        return -99000 if in_check else 0

    # In check: search all evasions; otherwise captures + promotions
    if in_check:
        candidates = moves
    else:
        candidates = [m for m in moves
                      if board.get(m[2], m[3]) is not None
                      or m[4] == 'ep'
                      or (board.get(m[0], m[1]) and board.get(m[0], m[1]).kind == 'P'
                          and (m[2] == 0 or m[2] == 7))]

    candidates = order_moves(board, candidates, color=color)

    for move in candidates:
        r0, c0, r1, c1, flag = move
        is_cap = board.get(r1, c1) is not None or flag == 'ep'
        # Skip losing captures (except when in check — must escape)
        if is_cap and not in_check and depth > 0 and is_losing_capture(board, move):
            continue
        nb = apply_move(board, r0, c0, r1, c1, flag)
        if nb is None: continue
        score = -quiescence(nb, -beta, -alpha, opp, depth + 1)
        if score >= beta: return beta
        if score > alpha: alpha = score

    return alpha

# ── Negamax with Alpha-Beta + TT + Null move + Check ext + Futility ─────────
NULL_REDUCTION = 2   # R=2 safer than R=3

FUTILITY_MARGIN = [0, 150, 300, 500]   # per depth 1/2/3

def negamax(board, depth, alpha, beta, color, deadline, ply=0, do_null=True):
    if time.time() > deadline:
        raise TimeoutError()

    opp        = 'b' if color == 'w' else 'w'
    alpha_orig = alpha

    # Transposition table probe
    zh = zobrist_hash(board, color)
    tt_val, tt_move = tt_probe(zh, depth, alpha, beta)
    if tt_val is not None:
        return tt_val, None

    in_check = is_in_check(board, color)

    # Check extension: go one ply deeper when in check
    if in_check:
        depth += 1

    if depth <= 0:
        val = quiescence(board, alpha, beta, color)
        return val, None

    # Null move pruning
    own_material = sum(PIECE_VALUE[p.kind] for r in range(8) for c in range(8)
                       if (p := board.get(r, c)) and p.color == color and p.kind != 'K')
    if (do_null and not in_check and depth >= NULL_REDUCTION + 1
            and not is_endgame(board) and own_material >= 1300
            and abs(beta) < 50000):
        null_board = board.copy()
        null_board.en_passant = None
        null_val, _ = negamax(null_board, depth - NULL_REDUCTION - 1,
                               -beta, -beta + 1, opp, deadline, ply + 1, do_null=False)
        null_val = -null_val
        if null_val >= beta:
            return beta, None   # No verify — R=2 is already conservative

    moves = legal_moves(board, color)
    if not moves:
        if in_check:
            return -99000 + ply, None
        return 0, None

    moves = order_moves(board, moves, ply=ply, tt_move=tt_move, color=color)
    best_move = None
    best_val  = -float('inf')

    # Static eval for futility pruning (only at low depths)
    static_eval = None
    if depth <= 3 and not in_check:
        raw_eval = evaluate(board)
        static_eval = raw_eval if color == 'w' else -raw_eval

    for i, move in enumerate(moves):
        r0, c0, r1, c1, flag = move
        is_capture  = board.get(r1, c1) is not None or flag == 'ep'
        is_promo    = board.get(r0, c0).kind == 'P' and (r1 == 0 or r1 == 7)
        is_killer   = ply < 64 and move in (_KILLERS[ply][0], _KILLERS[ply][1])

        # Futility pruning: skip quiet moves that can't raise alpha at low depth
        if (static_eval is not None and depth <= 3 and i > 0
                and not is_capture and not is_promo and not is_killer
                and not in_check and abs(alpha) < 50000):
            margin = FUTILITY_MARGIN[min(depth, 3)]
            if static_eval + margin <= alpha:
                continue

        # SEE pruning: skip badly losing captures at depth <= 2
        if is_capture and depth <= 2 and is_losing_capture(board, move):
            continue

        nb = apply_move(board, r0, c0, r1, c1, flag)
        if nb is None: continue

        gives_check = is_in_check(nb, opp)

        # Late Move Reduction — strictly quiet, non-tactical moves only
        reduction = 0
        if (i >= 4 and depth >= 3 and not is_capture and not is_promo
                and not in_check and not gives_check and not is_killer
                and not is_in_check(nb, color)):   # extra guard
            reduction = 1

        val, _ = negamax(nb, depth - 1 - reduction, -beta, -alpha, opp, deadline, ply + 1)
        val = -val

        # Re-search at full depth if LMR move beats alpha
        if reduction > 0 and val > alpha:
            val, _ = negamax(nb, depth - 1, -beta, -alpha, opp, deadline, ply + 1)
            val = -val

        if val > best_val:
            best_val  = val
            best_move = move
        if val > alpha:
            alpha = val
        if alpha >= beta:
            if not is_capture:
                update_killer(ply, move)
                _HISTORY[color][r0][c0][r1][c1] += depth * depth
            break

    # TT store
    if best_move:
        flag_str = ('upper' if best_val <= alpha_orig else
                    'lower' if best_val >= beta else 'exact')
        tt_store(zh, depth, flag_str, best_val, best_move)

    return best_val, best_move

# ── Iterative deepening ───────────────────────────────────────────────────────

def best_move_id(board, color, time_limit=TIME_LIMIT):
    """Iterative deepening with aspiration windows."""
    # Guaranteed fallback
    moves = legal_moves(board, color)
    if not moves:
        return None
    best = order_moves(board, moves, color=color)[0]

    # Clear killer table each new search
    for i in range(64):
        _KILLERS[i] = [None, None]

    deadline  = time.time() + time_limit
    prev_val  = 0
    WINDOW    = 50   # aspiration window

    for depth in range(1, 30):
        # Aspiration windows: narrow alpha/beta around previous score
        if depth >= 4:
            alpha = prev_val - WINDOW
            beta  = prev_val + WINDOW
        else:
            alpha = -float('inf')
            beta  =  float('inf')

        try:
            failures = 0
            while True:
                val, move = negamax(board, depth, alpha, beta, color, deadline)
                if val <= alpha:          # fail low — widen down
                    failures += 1
                    if failures >= 2:
                        alpha = -float('inf')
                    else:
                        alpha -= WINDOW * (2 ** failures)
                elif val >= beta:         # fail high — widen up
                    failures += 1
                    if failures >= 2:
                        beta = float('inf')
                    else:
                        beta += WINDOW * (2 ** failures)
                else:
                    break                 # score inside window — reliable
            # Only store result from a fully completed, non-failed search
            if move is not None:
                best     = move
                prev_val = val

        except TimeoutError:
            break                         # keep best from last *completed* depth

    return best

# ── Parse user input ──────────────────────────────────────────────────────────
def parse_move(text):
    text = text.strip().lower()
    if len(text) == 4:
        return text[:2], text[2:4], 'Q'
    if len(text) == 5 and text[4] in 'qrbn':
        return text[:2], text[2:4], text[4].upper()
    return None

# ── Replay viewer ────────────────────────────────────────────────────────────
def replay_game(history, human_color):
    """Step through every board state recorded during the game."""
    BOLD  = '\033[1m'
    RESET = '\033[0m'
    total = len(history)
    if total == 0:
        print("  No moves to replay.")
        return
    idx = 0
    print(f"\n  {'═'*52}")
    print(f"  REPLAY  —  {total} moves total")
    print(f"  ◀  prev    ▶  next    number = jump to move    q = quit")
    print(f"  {'═'*52}")
    while True:
        board, move_label = history[idx]
        board.display(perspective=human_color)
        print(f"  {BOLD}Move {idx}/{total-1}{RESET}  {move_label}")
        print(f"  [◀ b]  [▶ n/Enter]  [jump: 0-{total-1}]  [q quit]")
        try:
            raw = input("  → ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if raw in ('q', 'quit'):
            break
        elif raw in ('', 'n', 'f', '>'):
            idx = min(idx + 1, total - 1)
        elif raw in ('b', 'p', '<'):
            idx = max(idx - 1, 0)
        elif raw.isdigit():
            n = int(raw)
            if 0 <= n < total:
                idx = n
            else:
                print(f"  ✗  Enter a number between 0 and {total-1}")
        else:
            print("  n = next  b = back  number = jump  q = quit")


# ── Main game loop ────────────────────────────────────────────────────────────
def main():
    board = Board()
    opp   = {'w': 'b', 'b': 'w'}
    color_name = {'w': 'White', 'b': 'Black'}

    print("\n" + "═" * 52)
    print("          ♔  PYTHON CHESS  ♚")
    print("═" * 52)
    print("  Play against the computer!")
    print("  Choose your color:")
    print("    w = White (you move first)")
    print("    b = Black (computer moves first)")
    print("═" * 52)

    while True:
        choice = input("  Your color [w/b]: ").strip().lower()
        if choice in ('w', 'b'):
            human_color = choice
            break
        print("  Please enter w or b.")

    ai_color = opp[human_color]
    turn = 'w'

    net = _get_net()
    if _TORCH_OK:
        dev_name = str(_DEVICE).upper()
        if _DEVICE.type == 'cuda':
            dev_name = torch.cuda.get_device_name(0)
        print(f"\n  Neural network running on: {dev_name}")
    if net.games_trained == 0:
        print("  NN not trained yet — using classical engine.")
        print("  Run  python3 chess.py --train  to start learning.")
    else:
        wf = WEIGHTS_FILE.replace('.npz','.pt') if _TORCH_OK else WEIGHTS_FILE
        print(f"  NN active — trained on {net.games_trained} games.")
        print(f"  Weights: {wf}")
    print(f"\n  You play as {color_name[human_color]}. Good luck!")
    print("  Moves like  e2e4  |  promotion: e7e8q  |  resign to quit\n")

    # History stores (board_snapshot, label) for every position
    history = [(board.copy(), "Start")]

    while True:
        board.display(perspective=human_color)

        all_legal = legal_moves(board, turn)
        in_check  = is_in_check(board, turn)

        if not all_legal:
            if in_check:
                winner = color_name[opp[turn]]
                print(f"\n  ♛  CHECKMATE — {winner} wins!  ♛\n")
            else:
                print("\n  ½  STALEMATE — Draw!  ½\n")
            break

        if in_check:
            print(f"\n  ⚠  {color_name[turn]} is in CHECK!")

        # ── AI turn ──────────────────────────────────────────────────────
        if turn == ai_color:
            print(f"\n  ♟  Computer ({color_name[ai_color]}) is thinking…")
            move = best_move_id(board, ai_color)
            if move is None:
                print("  Computer has no moves — this shouldn't happen!")
                break
            r0, c0, r1, c1, flag = move
            nb = apply_move(board, r0, c0, r1, c1, flag)
            from_s = sq(r0, c0)
            to_s   = sq(r1, c1)
            print(f"  Computer plays: {from_s}{to_s}")
            board = nb
            if turn == 'b':
                board.full_moves += 1
            if is_in_check(board, opp[turn]):
                print(f"  ⚠  {color_name[opp[turn]]} is now in CHECK!")
            history.append((board.copy(), f"Computer: {from_s}{to_s}"))
            turn = opp[turn]
            continue

        # ── Human turn ────────────────────────────────────────────────────
        print(f"\n  Your turn ({color_name[human_color]})  —  move {board.full_moves}")
        try:
            raw = input("  → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Game interrupted. Goodbye!")
            sys.exit()

        if raw.lower() in ('quit', 'resign', 'q'):
            print(f"\n  You resigned. {color_name[ai_color]} wins!\n")
            break

        parsed = parse_move(raw)
        if parsed is None:
            print("  ✗  Invalid format. Use e.g. e2e4 or e7e8q")
            continue

        from_s, to_s, promo = parsed
        try:
            fr, fc = rc(from_s)
            tr, tc = rc(to_s)
        except (ValueError, IndexError):
            print("  ✗  Invalid square. Use a–h and 1–8.")
            continue

        matched = None
        for move in all_legal:
            r0, c0, r1, c1, flag = move
            if (r0, c0, r1, c1) == (fr, fc, tr, tc):
                matched = move
                break

        if matched is None:
            print("  ✗  Illegal move.")
            continue

        r0, c0, r1, c1, flag = matched
        new_board = apply_move(board, r0, c0, r1, c1, flag, promo)
        if new_board is None:
            print("  ✗  That move leaves your king in check.")
            continue

        board = new_board
        if turn == 'b':
            board.full_moves += 1
        if is_in_check(board, opp[turn]):
            print(f"\n  ⚠  {color_name[opp[turn]]} is now in CHECK!")
        history.append((board.copy(), f"You: {from_s}{to_s}"))
        turn = opp[turn]

    # ── Post-game replay offer ────────────────────────────────────────────
    print(f"  Game over — {len(history)-1} moves played.")
    try:
        ans = input("  Watch replay? [y/n]: ").strip().lower()
        if ans == 'y':
            replay_game(history, human_color)
    except (EOFError, KeyboardInterrupt):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# ── Self-play training loop ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def train_forever():
    """
    Run self-play games indefinitely, training the neural network after each one.
    Press Ctrl+C to stop — weights are saved automatically.
    Run with:  python3 chess.py --train
    """
    net    = _get_net()
    replay = _get_replay()

    # TRAIN_DEPTH is set at the top of the file
    MAX_MOVES   = 200    # draw after this many moves
    BATCH_EVERY = 5      # replay batch training every N games
    REPORT_EVERY= 10

    print("\n" + "═" * 56)
    print("  ♟  Chess Self-Play Trainer  ♟")
    print("═" * 56)
    if _TORCH_OK:
        dev_name = torch.cuda.get_device_name(0) if _DEVICE.type=='cuda' else str(_DEVICE).upper()
        print(f"  Device   : {dev_name}")
    else:
        print( "  Device   : CPU (numpy fallback — install PyTorch for GPU)")
    wf = WEIGHTS_FILE.replace('.npz','.pt') if _TORCH_OK else WEIGHTS_FILE
    print(f"  Weights  : {wf}")
    print(f"  Games    : {net.games_trained} trained so far")
    print(f"  Replay   : {len(replay.buf)} positions stored")
    print(f"  Depth    : {TRAIN_DEPTH} (per move)")
    print("  Press Ctrl+C to stop — progress saved after every game.")
    print("═" * 56 + "\n")

    opp_map  = {'w':'b','b':'w'}
    game_num = w_wins = b_wins = draws = 0

    try:
        while True:
            game_num += 1
            board    = Board()
            trainer  = TDTrainer(net)
            trainer.reset()
            turn     = 'w'
            outcome  = None
            positions = []

            for _ in range(MAX_MOVES):
                moves = legal_moves(board, turn)
                if not moves:
                    outcome = (-1.0 if is_in_check(board, turn) and turn=='w' else
                                1.0 if is_in_check(board, turn) else 0.0)
                    break

                move = best_move_id(board, turn, time_limit=2.0)
                if not move:
                    outcome = 0.0; break

                positions.append((encode_board(board, turn), turn))
                trainer.step(board, turn)

                r0,c0,r1,c1,flag = move
                board = apply_move(board, r0, c0, r1, c1, flag)
                if not board:
                    outcome = 0.0; break

                if board.half_moves >= 100:
                    outcome = 0.0; break

                if turn == 'b':
                    board.full_moves += 1
                turn = opp_map[turn]

            if outcome is None:
                outcome = 0.0

            trainer.finish(board, outcome)

            for enc, color in positions:
                signed = outcome if color == 'w' else -outcome
                replay.push(enc, signed)

            if game_num % BATCH_EVERY == 0:
                _batch_train(net, replay)
                net.save()
                replay.save()

            if outcome > 0:   w_wins += 1
            elif outcome < 0: b_wins += 1
            else:             draws  += 1

            if game_num % REPORT_EVERY == 0:
                t = w_wins + b_wins + draws
                print(f"  Game {net.games_trained:>5}  │  "
                      f"W {w_wins/t*100:4.0f}%  "
                      f"B {b_wins/t*100:4.0f}%  "
                      f"D {draws/t*100:4.0f}%  │  "
                      f"Replay {len(replay.buf):>6}")
                w_wins = b_wins = draws = 0

    except KeyboardInterrupt:
        net.save(); replay.save()
        print(f"\n  Stopped. Total games: {net.games_trained}")
        print(f"  Weights → {WEIGHTS_FILE}")


if __name__ == '__main__':
    if '--train' in sys.argv:
        train_forever()
    else:
        main()
