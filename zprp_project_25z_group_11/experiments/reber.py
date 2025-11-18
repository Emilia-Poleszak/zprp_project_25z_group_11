import random
import numpy as np

from torch import tensor, optim, Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from LRU_pytorch import LRU

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
        # TODO: check if padding is necessary
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
        no_success = 0
        for i, data in enumerate(loader):
            data = data[0]

            inputs = data[:-1, :].unsqueeze(0)
            targets = data[1:, :].argmax(dim=-1).unsqueeze(0)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # outputs = []
            # loss = 0
            # for t in range(inputs.size(1)):
            #     out = self.model(inputs[:, t, :])
            #     loss += self.loss_fn(out, targets[:, t])
            #     outputs.append(out)
            #
            # outputs = torch.stack(outputs, dim=1)

            no_success += (outputs.argmax(dim=-1).equal(targets))

            loss = self.loss_fn(outputs.reshape(-1, 7), targets.reshape(-1))
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        running_loss /= len(loader)

        print(f'Successful predictions: {format(no_success)} / {len(loader)}')
        return running_loss, no_success == len(loader)

    def eval(self):
        """
        Evaluates the model on validation set.
        :return:
        """
        self.model.eval()
        loader = DataLoader(self.validation_set, batch_size=1)
        total_tokens = 0
        correct_tokens = 0
        for batch in loader:
            batch = batch[0]
            inputs = batch[:-1, :].unsqueeze(0)
            targets = batch[1:, :].argmax(dim=-1).unsqueeze(0)

            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            total_tokens += targets.size(1)
            correct_tokens += (outputs.argmax(dim=-1).equal(targets))
        return correct_tokens / total_tokens

    def run(self):
        epoch = 0
        writer = SummaryWriter()
        while True:
            loss, success = self.train(epoch, writer)
            val_acc = self.eval()
            print('epoch {} loss: {}'.format(epoch, loss))
            print('Val acc: {}'.format(val_acc))
            self.model.eval()
            epoch += 1
            if success:
                break

    @staticmethod
    def string_to_one_hot(s: str, mapping: dict[str, int]) -> Tensor:
        encoded = np.zeros([len(s), len(mapping)], dtype=np.float32)
        for i, char in enumerate(s):
            encoded[i][mapping[char]] = 1
        return tensor(encoded)

# TODO: move this to other file after finishing experiment implementation (this is just for tests)
def main():
    model = LRU(in_features=7, out_features=7, state_features=64)
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    # writer = SummaryWriter()

    reber = ReberExperiment(model, optimizer)
    reber.setup()
    reber.run()

if __name__ == '__main__':
    main()