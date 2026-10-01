import torch
import torch.nn as nn
import torch.nn.functional as F

from ..JungleGame import NUM_PLANES


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class JungleNNet(nn.Module):
    """Input (N, 23, 9, 7). Policy: 4 logits per square (up, down, left, right),
    flattened so index = (r * 7 + c) * 4 + direction. Value: tanh scalar."""

    def __init__(self, game, args):
        super().__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        ch = args.num_channels

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(args.num_res_blocks)])

        self.policy = nn.Sequential(
            nn.Conv2d(ch, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 4, 1),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(ch, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(2 * self.board_x * self.board_y, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, s):
        x = self.body(self.stem(s))
        pi = self.policy(x).permute(0, 2, 3, 1).reshape(-1, self.action_size)
        v = self.value_fc(self.value_conv(x).flatten(1))
        return F.log_softmax(pi, dim=1), torch.tanh(v)
