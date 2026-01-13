from torch import optim, tensor, float32
import torch
import torch.nn as nn
from LRU_pytorch import LRU
from torch.utils.tensorboard import SummaryWriter

from zprp_project_25z_group_11.generators.components import Components
from zprp_project_25z_group_11.config import (MULTIPLICATION_LOGS_DIR,
                                              MULTIPLICATION_HIDDEN_SIZE,
                                              MULTIPLICATION_LEARNING_RATE,
                                              MULTIPLICATION_SEQUENCES,
                                              MULTIPLICATION_SEQUENCE_LENGTH,
                                              MULTIPLICATION_DATA_FILENAME)


class Multiplication(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 writer: SummaryWriter,
                 lr: float = MULTIPLICATION_LEARNING_RATE,
                 sequence_length: int = MULTIPLICATION_SEQUENCE_LENGTH,
                 value_range: tuple[float, float] = (0, 1.0),
                 alpha: float = 0.9):
        """
        Implementation of multiplication experiment.

        :param model: model inheriting from nn.Module: LSTM, LRU, GRU
        :param lr: learning rate for the RMSprop optimizer
        :param sequence_length: number of elements in data sequence
        :param value_range: range of values in data sequence
        :param alpha: smoothing constant for the RMSprop optimizer
        """
        super().__init__()

        self.model = model
        self.head = nn.Linear(model.hidden_size if hasattr(model, "hidden_size") else model.out_features, 1)
        self.optimizer = optim.RMSprop(list(self.model.parameters()) + list(self.head.parameters()),
                                       lr=lr,
                                       alpha=alpha)
        self.criterion = nn.MSELoss()
        self.generator = Components(length=sequence_length,
                                    value_range=value_range)
        self.writer = writer
        self.global_step = 0
        self._init_lstm_forget_bias()

    def _init_lstm_forget_bias(self):
        """
        Initializes forget gates in LSTM model to ensure long-term memory
        """
        if not isinstance(self.model, nn.LSTM):
            return

        hidden_size = self.model.hidden_size
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.data.zero_()
                param.data[hidden_size:2 * hidden_size] = 3.0

    def forward(self, seq):
        out = self.model(seq)

        # for LSTM and GRU use:
        if isinstance(out, tuple):
            out = out[0]

        last = out[:, -1, :]

        return self.head(last)

    def train_loop(self,
                   max_steps=300000,
                   threshold=0.04,
                   n=2000,
                   max_errors=13):
        """
        Trains model with generated data, with possible switch to
        getting data from .txt file (commented line 95).
        Training completes when in last 2000 predictions is 13 or fewer errors.
        Absolute error threshold is 0.04.
        """
        self.model.train()

        errors = []
        correct = []
        start_line_number = 0

        for step in range(1, max_steps + 1):
            self.global_step += 1

            # to generate new data use:
            seq, target = self.generator.generate_multiplication()

            # to get data from file use:
            # seq, target, start_line_number = self.generator.get_data(start_line_number, MULTIPLICATION_DATA_FILENAME)

            sequence = tensor(seq, dtype=float32).unsqueeze(0)
            targets = tensor([[target]], dtype=float32)

            self.optimizer.zero_grad()
            output = self(sequence)
            loss = self.criterion(output, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.head.parameters()), 1.0)
            self.optimizer.step()

            self.writer.add_scalar("Train/MSE", loss.item(), self.global_step)
            self.writer.flush()

            abs_error = torch.abs(output - targets).view(-1)
            tmp = (abs_error < threshold).float()
            errors.extend(abs_error.tolist())
            correct.extend(tmp.tolist())

            if len(errors) > n:
                errors = errors[-n:]
                correct = correct[-n:]

            if step % n == 0:
                avg_error = sum(errors[-n:]) / n
                c = int(sum(correct[-n:]))
                print(f"Step: {step} | Correct: {c}/2000 | Average error: {avg_error:.4f}")

            if len(correct) == n and sum(correct) > n - max_errors:
                print(f"\nTraining finished at step {step}")
                break

    def evaluate(self,
                 num_samples=MULTIPLICATION_SEQUENCES,
                 threshold=0.04):
        """
        Evaluates model in 2560 iterations.
        Absolute error threshold is 0.04.
        """
        self.model.eval()
        errors = []
        correct = 0

        with torch.no_grad():
            for _ in range(num_samples):
                seq, target = self.generator.generate_multiplication()

                sequence = tensor(seq, dtype=float32).unsqueeze(0)
                targets = tensor([[target]], dtype=float32)

                output = self(sequence)
                abs_error = torch.abs(output - targets).item()
                errors.append(abs_error)

                if abs_error < threshold:
                    correct += 1

        avg_error = sum(errors) / len(errors)
        accuracy = correct / num_samples * 100

        self.writer.add_scalar("Eval/avg_error", avg_error, self.global_step)
        self.writer.add_scalar("Eval/accuracy", accuracy, self.global_step)
        self.writer.flush()

        print(f"Accuracy: {accuracy:.2f}%, average error: {avg_error:.4f}")


if __name__ == '__main__':
    for i in range(10):
        writer = SummaryWriter(log_dir=MULTIPLICATION_LOGS_DIR / f"exp_{i}")

        # choose one model for experiment:
        model = nn.LSTM(input_size=2,
                        hidden_size=MULTIPLICATION_HIDDEN_SIZE,
                        num_layers=1,
                        batch_first=True)

        # model = LRU(in_features=2,
        #             out_features=1,
        #             state_features=MULTIPLICATION_HIDDEN_SIZE)

        # model = nn.GRU(input_size=2,
        #                hidden_size=MULTIPLICATION_HIDDEN_SIZE,
        #                num_layers=1,
        #                batch_first=True)

        multiplication = Multiplication(model, writer)
        multiplication.train_loop()
        multiplication.evaluate()

        print(f"Experiment {i} completed.")
        writer.close()
