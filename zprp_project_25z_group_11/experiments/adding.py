import random
import torch
import torch.nn as nn
import torch.optim as optim
from LRU_pytorch import LRU
from torch.utils.tensorboard import SummaryWriter
import argparse

from zprp_project_25z_group_11.generators.components import Components
from zprp_project_25z_group_11.config import (
    ADDING_LOGS_DIR, ADDING_LEARNING_RATE, ADDING_HIDDEN_SIZE, ADDING_EVAL_SEQUENCES, ADDING_SEQUENCE_LENGTH,
    ADDING_RANGE, ADDING_DATA_FILENAME, ADDING_THRESHOLD, ADDING_ALPHA)


class Adding(nn.Module):
    def __init__(self, model: nn.Module, rng: random.Random, sequence_length: int = 100,
                 value_range: tuple[float, float] = (-1.0, 1.0),
                 learning_rate: float = 1e-3, alpha: float = 0.9, writer: SummaryWriter | None = None):
        """
        The implementation of the forth Hochreiter experiment.
        It tests whether models can solve long time lag problems involving distributed, continues-valued representations.

        :param model: model inheriting from nn.Module
        :param sequence_length: number of elements in the generated sequences
        :param value_range: range of values in the generated sequences
        :param learning_rate: learning rate for the RMSprop optimizer
        :param alpha: smoothing constant for the RMSprop optimizer
        """
        super().__init__()

        self.model = model
        self.head = nn.Linear(model.hidden_size if hasattr(
            model, "hidden_size") else model.out_features, 1)
        self.optimizer = optim.RMSprop(list(self.model.parameters()) + list(self.head.parameters()), lr=learning_rate,
                                       alpha=alpha)
        self.loss_fn = nn.MSELoss()
        self.generator = Components(
            length=sequence_length, value_range=value_range, rng=rng)
        self.writer = writer
        self.global_step = 0
        self._init_lstm_forget_bias()

    def _init_lstm_forget_bias(self):
        """Initializes forget gates in LSTM model to ensure long-term memory
        """
        if not isinstance(self.model, nn.LSTM):
            return

        hidden_size = self.model.hidden_size
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.data.zero_()
                param.data[hidden_size:2 * hidden_size] = 3.0

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]  # LSTM and GRU case
        last = out[:, -1, :]
        return self.head(last)

    def train(self, max_steps=100000, threshold=0.04, data_mode="generate"):
        """Trains model with generated data. Training stops if average error is below 0.01 and 2000 most recent sequences
        were processed correctly (absolute error < 0.04).

        :param data_mode: indicator whether to generate data to train or use data from file
        :param threshold: maximal absolute error for sequence to be processed correctly
        :param max_steps: maximum number of iterations
        """
        self.model.train()

        recent_errors = []
        recent_correct = []
        window = 2000
        start_line_number = 0

        for step in range(1, max_steps + 1):
            self.global_step += 1

            if data_mode == "generate":
                seq, target = self.generator.generate_adding()
            else:
                seq, target, start_line_number = self.generator.get_data(
                    start_line_number, ADDING_DATA_FILENAME)

            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor([[target]], dtype=torch.float32)

            self.optimizer.zero_grad()
            output = self(x)
            loss = self.loss_fn(output, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.head.parameters()), 1.0)
            self.optimizer.step()

            if self.writer is not None:
                self.writer.add_scalar(
                    "Train/MSE", loss.item(), self.global_step)
                self.writer.flush()

            abs_error = torch.abs(output - y).view(-1)
            correct = (abs_error < threshold).float()
            recent_errors.extend(abs_error.tolist())
            recent_correct.extend(correct.tolist())

            if len(recent_errors) > window:
                recent_errors = recent_errors[-window:]
                recent_correct = recent_correct[-window:]

            if step % 2000 == 0:
                n = min(1000, len(recent_errors))
                avg = sum(recent_errors[-n:]) / n
                corr = sum(recent_correct[-n:])
                print(
                    f"Step {step:6d} | " f"Loss {loss.item():.4f} | " f"AvgErr {avg:.4f} | " f"Correct {int(corr)}/{n}")

            if len(recent_correct) == window and sum(recent_correct) == window:
                avg_recent_error = sum(recent_errors) / window

                if avg_recent_error < 0.01:
                    print(f"\n*** SUCCESS at step {step}! ***")
                    break

    def evaluate_model(self, num_samples=2560, threshold=0.04):
        """Evaluates model on validation set.

        :param num_samples: validation set size
        :param threshold: maximal absolute error for sequence to be processed correctly
        :return: average error and accuracy
        """
        self.model.eval()
        errors = []
        correct = 0

        with torch.no_grad():
            for _ in range(num_samples):
                seq, target = self.generator.generate_adding()

                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                y = torch.tensor([[target]], dtype=torch.float32)

                output = self(x)
                err = torch.abs(output - y).item()

                errors.append(err)
                if err < threshold:
                    correct += 1

        avg_error = sum(errors) / len(errors)
        accuracy = correct / num_samples

        if self.writer is not None:
            self.writer.add_scalar(
                "Eval/avg_error", avg_error, self.global_step)
            self.writer.add_scalar("Eval/accuracy", accuracy, self.global_step)
            self.writer.flush()

        print("\n=== Evaluation ===")
        print(f"Avg error: {avg_error:.4f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")


def choose_model(input_model: str):
    model = None
    if input_model == "LSTM":
        model = nn.LSTM(input_size=2, hidden_size=ADDING_HIDDEN_SIZE,
                        num_layers=1, batch_first=True)
    elif input_model == "GRU":
        model = nn.GRU(input_size=2, hidden_size=ADDING_HIDDEN_SIZE,
                       num_layers=1, batch_first=True)
    elif input_model == "LRU":
        model = LRU(in_features=2, out_features=ADDING_HIDDEN_SIZE,
                    state_features=ADDING_HIDDEN_SIZE)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["LSTM", "GRU", "LRU"])
    parser.add_argument("--data", type=str, required=True,
                        choices=["generate", "file"])
    parser.add_argument("--rng", type=random.Random, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        writer = SummaryWriter(log_dir=ADDING_LOGS_DIR / f"exp_adding")

        model = choose_model(args.model)

        adding = Adding(model, args.rng, sequence_length=ADDING_SEQUENCE_LENGTH, value_range=ADDING_RANGE,
                        learning_rate=ADDING_LEARNING_RATE, alpha=ADDING_ALPHA, writer=writer)
        adding.train(data_mode=args.data, threshold=ADDING_THRESHOLD)
        adding.evaluate_model(
            num_samples=ADDING_EVAL_SEQUENCES, threshold=ADDING_THRESHOLD)
        print(f"Experiment completed.")
        writer.close()
    except Exception as e:
        print("Error detected:", e)
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
