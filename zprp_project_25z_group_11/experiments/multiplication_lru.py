from torch import optim, tensor, float32
import torch
import torch.nn as nn
from LRU_pytorch import LRU
from torch.nn.utils.rnn import pad_sequence

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
        i = 0
        errors = []
        start_line_number = 0

        self.train()

        while True:
            i += 1

            # generate new data:
            # seq, target = self.generator.generate_multiplication()

            # get data from file:
            # seq, target, start_line_number = self.generator.get_data(start_line_number)
            seq_batch = []
            target_batch = []
            seq_lengths = []

            for _ in range(8):
                seq, target = self.generator.generate_multiplication()
                seq_batch.append(torch.tensor(seq, dtype=torch.float32))
                seq_lengths.append(len(seq))
                target_batch.append(target)

            seq_lengths = torch.tensor(seq_lengths)
            target_batch = torch.tensor(target_batch, dtype=torch.float32).unsqueeze(1)

            seq_padded = pad_sequence(seq_batch, batch_first=True)

            self.optimizer.zero_grad()
            output = self(seq_padded, seq_lengths)
            loss = self.criterion(output, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            abs_error = torch.abs(output - target_batch).view(-1)
            tmp = (abs_error > 0.04).float()
            errors.extend(tmp.tolist())
            errors = errors[-2000:]

            # training stops when in last 2000 predictions there are 13 or fewer errors
            if len(errors) == 2000 and sum(errors) <= 13:
                break

            if i % 100 == 0:
                avg_error = sum(errors[-100:]) / 100
                err = sum(errors)
                print(f"iter: {i}, errors: {err}/2000, loss: {avg_error:.4f}")

    @torch.no_grad()
    def evaluate(self):
        self.eval()
        errors = []
        correct = 0

        for _ in range(2000):
            seq, target = self.generator.generate_multiplication()

            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            length = torch.tensor([len(seq)])

            output = self(seq_tensor, length)

            target_tensor = torch.tensor([[target]], dtype=torch.float32)
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