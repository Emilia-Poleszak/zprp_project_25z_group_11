from torch import optim, tensor, float32
import torch
import torch.nn as nn

from zprp_project_25z_group_11.generators.components import Components

class Multiplication_GRU(nn.Module):
    """
    Implementation of multiplication experiment with GRU architecture.
    """
    def __init__(self):
        super().__init__()

        hidden_dim = 128

        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.generator = Components(min_no_samples=100, value_range=(0.0, 1.0))
        self.criterion = nn.MSELoss()


    def forward(self, x, lengths):
        out, _ = self.gru(x)
        last = out[torch.arange(x.size(0)), lengths - 1]
        return self.head(last)

    def train_loop(self):
        """
        Trains GRU model with generated data, with possible switch to
        getting data from .txt file (commented line 54).
        Training stops when in last 2000 predictions is 13 or fewer errors.
        Absolute error threshold is 0.04.
        """
        i = 0
        correct = []
        errors = []
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

            # for _ in range(8):
            #     seq, target = self.generator.generate_multiplication()
            #     seq_batch.append(seq)
            #     target_batch.append(target)

            sequence = tensor(seq, dtype=float32).unsqueeze(0)
            targets = tensor([[target]], dtype=float32)

            self.optimizer.zero_grad()

            output, _ = self.gru(sequence)
            last_output = output[:, -1, :]
            pred = self.head(last_output)
            loss = self.criterion(pred, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

            abs_error = torch.abs(pred - targets).item()
            errors.append(abs_error)
            correct.append(abs_error < 0.04)
            correct = correct[-2000:]

            # training stops when in last 2000 predictions there are 13 or fewer errors
            if len(correct) == 2000 and (2000 - sum(correct)) <= 13:
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

            output, _ = self.gru(seq)
            target_tensor = torch.tensor([[target]], dtype=torch.float32)
            abs_error = torch.abs(output - target_tensor).item()

            errors.append(abs_error)
            if abs_error < 0.04:
                correct += 1

        avg_error = sum(errors) / len(errors)
        accuracy = correct / 2000

        print(f"Accuracy: {accuracy * 100:.2f}%, error: {avg_error:.4f}")


if __name__=="__main__":
    model = Multiplication_GRU()
    model.train_loop()
    model.evaluate()