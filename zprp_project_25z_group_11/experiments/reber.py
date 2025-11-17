from torch import tensor, optim, Tensor
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import random_split, TensorDataset
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
        n_val = len(sequences) - n_train

        batch_one_hot = [self.string_to_one_hot(seq[:-1], REBER_ALPHABET) for seq in sequences]
        # TODO: decide if max sequence length should be constant for every experiment exec
        batch_padded = pad_sequence(batch_one_hot, batch_first=True, padding_value=0)
        one_hot_dataset = TensorDataset(batch_padded)

        self.training_set, self.validation_set = random_split(one_hot_dataset, (n_train, n_val))

    def train_epoch(self, epoch: int):
        """
        Runs one training epoch.
        :return:
        """
        ...


    def eval(self):
        """
        Evaluates the model on validation set.
        :return:
        """
        ...

    def run(self):
        ...

    @staticmethod
    def string_to_one_hot(s: str, mapping: dict[str, int]) -> Tensor:
        encoded = np.zeros([len(s), len(mapping)])
        for i, char in enumerate(s):
            encoded[i][mapping[char]] = 1
        return tensor(encoded)

# TODO: move this to other file after finishing experiment implementation (this is just for tests)
def main():
    model = LRU(in_features=7, out_features=7, state_features=0)
    optimizer = optim.Adam(model.parameters())
    reber = ReberExperiment(model, optimizer)
    reber.setup()

if __name__ == '__main__':
    main()