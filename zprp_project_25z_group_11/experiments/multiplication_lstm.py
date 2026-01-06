from torch import optim
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

from zprp_project_25z_group_11.generators.components import Components

class Multiplication_LSTM(nn.Module):
    """
    Implementation of multiplication experiment with LSTM architecture.
    """
    def __init__(self):
        super().__init__()

        hidden_dim = 128

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.generator = Components(min_no_samples=100, value_range=(0.0, 1.0))
        self.criterion = nn.MSELoss()

    def _init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                h = self.hidden_size
                param.data.zero_()
                param.data[h:2 * h] = 3.0

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        last_out = out[torch.arange(x.size(0)), lengths - 1]

        return self.head(last_out)

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
    model = Multiplication_LSTM()
    model.train_loop()
    model.evaluate()