import random

import torch
from numpy.random import shuffle
from torch import tensor, optim, Tensor
import torch.nn as nn

from torch.utils.data import random_split, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from LRU_pytorch import LRU
import numpy as np

from zprp_project_25z_group_11.config import RAW_DATA_DIR, TRAIN_TEST_SPLIT, RAW_REBER_DATA_FILENAME, REBER_ALPHABET


class ReberExperiment:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer):
        """
        The implementation of the first Hochreiter experiment.
        It is a standard benchmark test for recurrent nets, whose task is to learn the Embedded Reber Grammar.

        :param model: nn.Module:
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.training_set = None
        self.validation_set = None

    def setup(self) -> None:
        """
        Loads generated data and splits it into training and validation sets.
        :return:
        """
        try:
            with open(RAW_DATA_DIR / RAW_REBER_DATA_FILENAME, 'r') as f:
                sequences = f.readlines()
            assert len(sequences) > 0
        except Exception as e:
            raise Exception(f'failed to load sequences from {RAW_DATA_DIR / RAW_REBER_DATA_FILENAME} file: {e}')

        n_train = int(len(sequences) * TRAIN_TEST_SPLIT)

        batch_one_hot = [self.string_to_one_hot(seq[:-1], REBER_ALPHABET) for seq in sequences]
        random.shuffle(batch_one_hot)
        # TODO: decide if max sequence length should be constant for every experiment exec
        # batch_padded = pad_sequence(batch_one_hot, batch_first=True, padding_value=0)

        self.training_set = batch_one_hot[:n_train]
        self.validation_set = batch_one_hot[n_train:]

    def train(self, epoch: int, tb_writer: SummaryWriter):
        """
        Runs one training epoch.
        :return:  the average per-batch loss for the last 1000 batches
        """
        if self.training_set is None:
            raise Exception('No training data set.')
        loader = DataLoader(self.training_set, shuffle=True, batch_size=1)
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(loader):
            data = data[0]
            # print(f'Size: {data.size()}')

            inputs = data[:-1, :].unsqueeze(0)
            targets = data[1:, :].argmax(dim=-1).unsqueeze(0)
            # print(f'Inputs: {inputs.shape}')
            # print(f'Targets: {targets.shape}')

            self.optimizer.zero_grad()

            outputs = []
            for t in range(inputs.size(1)):
                out = self.model(inputs[:, t, :])
                outputs.append(out)

            outputs = torch.stack(outputs, dim=1)
            # print(f'Outputs: {outputs}')

            loss = self.loss_fn(outputs.reshape(-1, 7), targets.reshape(-1))
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(self.training_set) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def eval(self):
        """
        Evaluates the model on validation set.
        :return:
        """
        ...

    def run(self):
        epoch = 0
        writer = SummaryWriter()
        while True:
            loss = self.train(epoch, writer)
            print('epoch {} loss: {}'.format(epoch, loss))
            epoch += 1
            if epoch >= 100:
                break


    @staticmethod
    def string_to_one_hot(s: str, mapping: dict[str, int]) -> Tensor:
        encoded = np.zeros([len(s), len(mapping)], dtype=np.float32)
        for i, char in enumerate(s):
            encoded[i][mapping[char]] = 1
        return tensor(encoded)

# TODO: move this to other file after finishing experiment implementation (this is just for tests)
def main():
    model = LRU(in_features=7, out_features=7, state_features=28)
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter()

    reber = ReberExperiment(model, optimizer)
    reber.setup()
    reber.run()

if __name__ == '__main__':
    main()