import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import AverageMeter, dotdict
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .JungleNNet import JungleNNet

args = dotdict({
    'lr': 0.001,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 64,
    'num_res_blocks': 6,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.nnet = JungleNNet(game, args)
        self.action_size = game.getActionSize()
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.nnet.to(self.device)

    def train(self, examples):
        """examples: list of (canonical board, pi, v)."""
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = max(1, len(examples) // args.batch_size)
            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in ids]))
                x = torch.from_numpy(self.game.encode(np.array(boards))).to(self.device)
                target_pis = torch.from_numpy(np.array(pis, dtype=np.float32)).to(self.device)
                target_vs = torch.from_numpy(np.array(vs, dtype=np.float32)).to(self.device)

                out_pi, out_v = self.nnet(x)
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size(0)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), x.size(0))
                v_losses.update(l_v.item(), x.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """board: one canonical board. Returns (policy over actions, value in [-1, 1])."""
        x = torch.from_numpy(self.game.encode(board[None])).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(x)
        return torch.exp(pi)[0].cpu().numpy(), float(v[0, 0].item())

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        os.makedirs(folder, exist_ok=True)
        torch.save({'state_dict': self.nnet.state_dict()}, os.path.join(folder, filename))

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path {}".format(filepath))
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        except TypeError:   # older PyTorch without weights_only
            checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
