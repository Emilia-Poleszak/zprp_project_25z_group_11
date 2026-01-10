from torch import optim, tensor, float32
import torch
import torch.nn as nn

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

    def forward(self, x, lengths):
        out, _ = self.lstm(x)
        last_out = out[torch.arange(x.size(0)), lengths - 1]
        return self.head(last_out)

    def train_loop(self):
        """
        Trains LSTM model with generated data, with possible switch to
        getting data from .txt file (commented line 53).
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
            output = self.lstm(sequence)
            last_output = output[:, -1, :]
            pred = self.head(last_output)
            loss = self.criterion(pred, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

            abs_error = torch.abs(pred - targets).item()
            correct.append(abs_error < 0.04)
            errors.append(abs_error)
            errors = errors[-2000:]

            # training stops when in last 2000 predictions there are 13 or fewer errors
            if len(errors) == 2000 and (2000 - sum(errors)) <= 13:
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