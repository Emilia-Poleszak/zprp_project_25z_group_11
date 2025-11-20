from torch import tensor, float32, optim
import torch.nn as nn

from LRU_pytorch import LRU

from zprp_project_25z_group_11.generators.components import Components

# TODO: modify so it's reproducible (it works dynamically now nut it has to read from file too)
class AddingExperiment:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, min_no_samples: int):
        """
        The implementation of the forth Hochreiter experiment.

        :param model: nn.Module:
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()
        self.min_no_samples = min_no_samples
        self.generator = Components(min_no_samples=self.min_no_samples, value_range=(-1.0, 1.0))

    def train(self):

        correct_streak = 0
        total_loss = 0
        step = 0
        while True:
            seq, target = self.generator.generate()
            seq = tensor(seq, dtype=float32).unsqueeze(1)  # (T,1,D)
            target = tensor(target, dtype=float32)

            self.optimizer.zero_grad()
            output = self.model(seq)
            loss = self.loss_fn(output[-1], target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            predicted = output[-1].item()
            if abs(predicted - target.item()) < 0.04:
                correct_streak += 1
            else:
                correct_streak = 0

            avg_loss = total_loss / (step + 1)
            print(f"{step} Error {abs(predicted - target.item())}, Avg loss: {avg_loss}, correct streak: {correct_streak}...")
            if avg_loss < 0.01 and correct_streak == 2000:
                print(f"Training converged at step {step}.")
                break
            step += 1

    def eval(self):
        """
        Evaluates the model on validation set.
        :return: number of correct predictions
        """
        self.model.eval()
        correct_pred = 0
        for _ in range(2560):
            seq, target = self.generator.generate()
            target = tensor(target, dtype=float32)
            output = self.model(seq)
            if abs(output[-1].item() - target.item()) < 0.04:
                correct_pred += 1
        return correct_pred


    def run(self):
        self.train()
        self.model.eval()

# TODO: move this to other file after finishing experiment implementation (this is just for tests)
def main():
    model = LRU(in_features=2, out_features=1, state_features=4)
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    # writer = SummaryWriter()

    adding = AddingExperiment(model, optimizer, 100)
    adding.run()

if __name__ == '__main__':
    main()