from torch import optim, tensor, float32
import torch
import torch.nn as nn
from LRU_pytorch import LRU

from zprp_project_25z_group_11.generators.components import Components


class Multiplication(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_dim = 32

        self.lru = LRU(
            in_features=2,
            out_features=hidden_dim,
            state_features=hidden_dim
        )
        self.head = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.generator = Components(min_no_samples=100, value_range=(0.0, 1.0))
        self.criterion = nn.SmoothL1Loss(beta=0.04)

    def forward(self, seq):
        # seq: (1, T, 2)
        out = self.lru(seq)          # (1, T, hidden)
        last = out[:, -1, :]         # (1, hidden)
        return self.head(last)       # (1, 1)

    def train_loop(self):
        i = 0
        errors = []

        while True:
            i += 1

            seq, target = self.generator.generate()
            seq = tensor(seq, dtype=float32).unsqueeze(0)
            target = tensor([[target]], dtype=float32)

            self.optimizer.zero_grad()
            output = self(seq)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            abs_error = torch.abs(output - target).item()

            errors.append(abs_error > 0.04)
            errors = errors[-2000:]

            if len(errors) == 2000 and sum(errors) <= 13:
                break
            if i % 100 == 0:
                print(f"iter: {i}, errors: {sum(errors)}/2000, loss: {loss.item():.4f}")


if __name__ == "__main__":
    m = Multiplication()
    m.train_loop()
