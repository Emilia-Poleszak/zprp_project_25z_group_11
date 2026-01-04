import random
import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from LRU_pytorch import LRU

from zprp_project_25z_group_11.config import (
    RAW_DATA_DIR,
    REBER_TRAIN_TEST_SPLIT,
    RAW_REBER_DATA_FILENAME,
    REBER_ALPHABET,
    REBER_LOGS_DIR, REBER_LEARNING_RATE, REBER_HIDDEN_SIZE)


class ReberExperiment:
    def __init__(self, model: nn.Module, learning_rate: float = 0.1):
        """
        The implementation of the first Hochreiter experiment.
        It is a standard benchmark test for recurrent nets, whose task is to learn the Embedded Reber Grammar.

        :param model: model inheriting from nn.Module
        :param learning_rate: learning rate for the SGD optimizer
        """
        self.grammar = {0: [('B', 1)],
                        1: [('T', 2), ('P', 3)],
                        2: [('S', 2), ('X', 4)],
                        3: [('T', 3), ('V', 5)],
                        4: [('X', 3), ('S', 6)],
                        5: [('P', 4), ('V', 6)],
                        6: [('E', 7)]}

        self.model = model
        self.output_layer = nn.Linear(model.hidden_size if hasattr(model, "hidden_size") else model.out_features, 7)
        self.optimizer = torch.optim.SGD(list(model.parameters()) + list(self.output_layer.parameters()), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        self.dataset = None
        self.training_set = None
        self.validation_set = None

    def setup(self) -> None:
        """
        Loads generated data and splits it into training and validation sets.
        """
        try:
            with open(RAW_DATA_DIR / RAW_REBER_DATA_FILENAME, 'r') as f:
                sequences = f.readlines()
            assert len(sequences) > 0
        except Exception as e:
            raise Exception(f'failed to load sequences from {RAW_DATA_DIR / RAW_REBER_DATA_FILENAME} file: {e}')

        n_train = int(len(sequences) * REBER_TRAIN_TEST_SPLIT)

        batch_one_hot = [self.string_to_one_hot(seq[:-1], REBER_ALPHABET) for seq in sequences]
        random.shuffle(batch_one_hot)
        # TODO: check if padding is necessary
        # batch_padded = pad_sequence(batch_one_hot, batch_first=True, padding_value=0)

        self.dataset = batch_one_hot
        self.training_set = batch_one_hot[:n_train]
        self.validation_set = batch_one_hot[n_train:]

    def train(self):
        """
        Runs one training epoch.
        :return: the average per-batch loss for the last 1000 batches
        """
        if self.training_set is None:
            raise Exception('No training data set.')
        loader = DataLoader(self.training_set, batch_size=1, shuffle=True)
        total_loss = 0.

        self.model.train()
        for data in loader:
            self.optimizer.zero_grad()
            # data = data[0]

            inputs = data[:, :-1, :]
            targets = data[:, 1:, :].argmax(dim=-1).view(-1)

            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # LSTM case
            outputs = self.output_layer(outputs.squeeze(0))

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(self.training_set)
        return avg_loss

    def eval(self) -> (float, int):
        """
        Evaluates the model on the both training and validation set.
        :return: Achieved score and number of samples with successfully predicted letters.
        """
        self.model.eval()
        success_count = 0
        inv_map = {v: k for k, v in REBER_ALPHABET.items()}

        with torch.no_grad():
            for data in self.dataset:
                seq_tensor = data  # (seq_len, 7)
                seq_indices = seq_tensor.argmax(dim=-1).tolist()
                seq_chars = [inv_map[idx] for idx in seq_indices]

                reber_state = 0
                gate = seq_chars[1]
                is_sequence_valid = True

                for t in range(len(seq_chars) - 1):
                    current_char = seq_chars[t]
                    current_history = seq_tensor[:t + 1].unsqueeze(0)

                    output = self.model(current_history)
                    if isinstance(output, tuple): output = output[0]

                    last_step_output = output[:, -1, :]  # (1, hidden)
                    logits = self.output_layer(last_step_output).squeeze(0)  # (7,)

                    allowed_next = []
                    if t == 0:
                        allowed_next = ['T', 'P']
                    elif reber_state < 7:
                        transitions = self.grammar[reber_state]
                        allowed_next = [trans[0] for trans in transitions]
                        next_char_in_seq = seq_chars[t + 1]
                        next_state = None
                        for char, new_state in transitions:
                            if char == next_char_in_seq:
                                next_state = new_state
                                break
                        if next_state is not None: reber_state = next_state

                    if reber_state == 7:
                        if current_char == 'E':
                            allowed_next = [gate]
                        elif current_char == gate:
                            allowed_next = ['E']

                    if not allowed_next: continue

                    valid_indices = [REBER_ALPHABET[c] for c in allowed_next]
                    invalid_indices = [i for i in range(len(REBER_ALPHABET)) if i not in valid_indices]

                    min_valid = torch.min(logits[valid_indices]).item()
                    if len(invalid_indices) > 0:
                        max_invalid = torch.max(logits[invalid_indices]).item()
                        if min_valid <= max_invalid:
                            is_sequence_valid = False
                            break

                if is_sequence_valid:
                    success_count += 1

        score = success_count / len(self.dataset)
        return score, success_count

    def run(self, summary_writer: SummaryWriter):
        """
        Runs a single Embedded Reber Grammar experiment.
        :param summary_writer: writer for loss and score documentation
        """
        epoch = 0
        while True:
            epoch += 1

            loss = self.train()
            score, success_count = self.eval()

            summary_writer.add_scalar('Loss', loss, epoch)
            summary_writer.add_scalar("Condition score", score, epoch)
            print(f"Epoch {epoch}: loss = {loss:.4f}, score = {score:.4f}")

            if score == 1:
                break
        summary_writer.flush()

    @staticmethod
    def string_to_one_hot(s: str, mapping: dict[str, int]) -> torch.Tensor:
        """
        Converts a string to a list of one-hot vectors based on a given mapping.
        :param s: converted string
        :param mapping: dictionary of possible letters and their order
        :return: one-hot vectors for each letter in a given string
        """
        encoded = np.zeros([len(s), len(mapping)], dtype=np.float32)
        for i, char in enumerate(s):
            encoded[i][mapping[char]] = 1
        return torch.tensor(encoded)


if __name__ == '__main__':
    for i in range(30):
        writer = SummaryWriter(log_dir=REBER_LOGS_DIR / f"exp_{i}")

        # choose model
        model = LRU(in_features=7, out_features=7, state_features=REBER_HIDDEN_SIZE)
        # model = nn.LSTM(input_size=7, hidden_size=REBER_HIDDEN_SIZE, num_layers=1, batch_first=True)
        # model = nn.GRU(input_size=7, hidden_size=REBER_HIDDEN_SIZE, num_layers=1, batch_first=True)

        reber = ReberExperiment(model, learning_rate=REBER_LEARNING_RATE)
        reber.setup()
        reber.run(writer)

        print(f"Experiment {i} completed.")
        writer.close()