from torch import optim, tensor, float32
import torch
import torch.nn as nn
from LRU_pytorch import LRU

from zprp_project_25z_group_11.generators.components import Components

class Multiplication_LRU(nn.Module):
    """
    Implementation of multiplication experiment with LRU architecture.
    """
    def __init__(self):
        super().__init__()

        hidden_dim = 128

        self.lru = LRU(
            in_features=2,
            out_features=hidden_dim,
            state_features=hidden_dim
        )
        self.head = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.generator = Components(min_no_samples=100, value_range=(0.0, 1.0))
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)

    def forward(self, seq, l):
        b, t, _ = seq.shape
        device = seq.device
        mask = (torch.arange(t, device=device)[None, :] < l[:, None]).unsqueeze(-1)

        out = self.lru(seq)
        out = out * mask
        last = out[torch.arange(b, device=device), l - 1]

        return self.head(last)

    def train_loop(self):
        """
        Trains LRU model with generated data, with possible switch to
        getting data from .txt file (commented line 61).
        Training stops when in last 2000 predictions is 13 or fewer errors.
        Absolute error threshold is 0.04.
        """
        i = 0
        errors = []
        correct = []
        start_line_number = 0

        self.train()

        while True:
            i += 1

            # generate new data:
            seq, target = self.generator.generate_multiplication()

            # get data from file:
            # seq, target, start_line_number = self.generator.get_data(start_line_number)
            # seq_batch = []
            # target_batch = []
            # seq_lengths = []
            #
            # for _ in range(8):
            #     seq, target = self.generator.generate_multiplication()
            #     seq_batch.append(torch.tensor(seq, dtype=torch.float32))
            #     seq_lengths.append(len(seq))
            #     target_batch.append(target)

            sequence = tensor(seq, dtype=float32).unsqueeze(0)
            targets = tensor([[target]], dtype=float32)

            self.optimizer.zero_grad()

            output = self.lru(sequence)
            last_output = output[:, -1, :]
            pred = self.head(last_output)
            loss = self.criterion(pred, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            abs_error = torch.abs(pred - targets).item()
            correct.append(abs_error < 0.04)
            errors.append(abs_error)
            errors = errors[-2000:]

            # training stops when in last 2000 predictions there are 13 or fewer errors
            if len(errors) == 2000 and (2000 - sum(correct)) <= 13:
                break

            if i % 100 == 0:
                avg_error = sum(errors) / len(errors)
                c = sum(correct)
                print(f"iter: {i}, correct: {c}/2000, loss: {avg_error:.4f}")

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluates model in 2000 iterations.
        Absolute error threshold is 0.04.
        """
        self.eval()
        errors = []
        correct = 0

        for _ in range(2000):
            seq, target = self.generator.generate_multiplication()

            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([[target]], dtype=torch.float32)
            length = torch.tensor([len(seq)])

            output = self.lru(seq_tensor, length)

            abs_error = torch.abs(output - target_tensor).item()
            errors.append(abs_error)
            if abs_error < 0.04:
                correct += 1

        avg_error = sum(errors) / len(errors)
        accuracy = correct / 2000

        print(f"Accuracy: {accuracy * 100:.2f}%, error: {avg_error:.4f}")


if __name__=="__main__":
    model = Multiplication_LRU()
    model.train_loop()
    model.evaluate()