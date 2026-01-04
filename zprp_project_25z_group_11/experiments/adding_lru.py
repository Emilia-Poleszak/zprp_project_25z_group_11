import random
import torch
import torch.nn as nn
from LRU_pytorch import LRU
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from zprp_project_25z_group_11.generators.components import Components

class AddingLRU(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, state_size=128):
        super().__init__()

        self.lru = LRU(in_features=input_size, out_features=hidden_size, state_features=state_size)

        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        device = x.device

        out = self.lru(x)

        mask = (torch.arange(T, device=device)[None, :] < lengths[:, None])

        mask = mask.unsqueeze(-1)
        out = out * mask

        last_out = out[torch.arange(B, device=device), lengths - 1]

        return self.head(last_out)

def train_lru(max_steps=300_000, batch_size=8, min_seq_len=100, hidden_size=128, state_size=128):
    model = AddingLRU(input_size=2, hidden_size=hidden_size, state_size=state_size)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    generator = Components(min_no_samples=min_seq_len, value_range=(-1.0, 1.0))

    recent_errors = []
    recent_correct = []
    window = 2000

    for step in range(1, max_steps + 1):

        seq_tensors = []
        lengths = []
        targets = []

        for _ in range(batch_size):
            seq, target = generator.generate()
            seq_tensors.append(torch.tensor(seq, dtype=torch.float32))
            lengths.append(len(seq))
            targets.append(target)

        lengths = torch.tensor(lengths)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        seq_padded = pad_sequence(seq_tensors, batch_first=True)

        optimizer.zero_grad()

        output = model(seq_padded, lengths)
        loss = criterion(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        abs_error = torch.abs(output - targets).view(-1)
        correct = (abs_error < 0.04).float()

        recent_errors.extend(abs_error.tolist())
        recent_correct.extend(correct.tolist())

        if len(recent_errors) > window:
            recent_errors = recent_errors[-window:]
            recent_correct = recent_correct[-window:]

        if step % 200 == 0:
            n = min(1000, len(recent_errors))
            avg = sum(recent_errors[-n:]) / n
            corr = sum(recent_correct[-n:])

            print(
                f"Step {step:6d} | "
                f"Loss: {loss.item():.4f} | "
                f"AvgErr: {avg:.4f} | "
                f"Correct: {int(corr)}/{n}"
            )

            # evaluate_model(model)

        if len(recent_errors) == window and sum(recent_correct) == window:
            print(f"\n*** SUCCESS at step {step}! ***")
            break

    return model

@torch.no_grad()
def evaluate_model(model, num_samples=2000, min_seq_len=100, threshold=0.04):
    model.eval()
    generator = Components(min_no_samples=min_seq_len, value_range=(-1.0, 1.0))

    errors = []
    correct = 0

    for _ in range(num_samples):
        seq, target = generator.generate()

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        length = torch.tensor([len(seq)])

        output = model(seq_tensor, length)

        target_tensor = torch.tensor([[target]], dtype=torch.float32)
        abs_error = torch.abs(output - target_tensor).item()

        errors.append(abs_error)
        if abs_error < threshold:
            correct += 1

    avg_error = sum(errors) / len(errors)
    accuracy = correct / num_samples

    print("\n=== Evaluation ===")
    print(f"Avg error: {avg_error:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    model.train()
    return avg_error, accuracy

if __name__ == "__main__":
    trained_model = train_lru()

    evaluate_model(trained_model, num_samples=2000, threshold=0.04)
