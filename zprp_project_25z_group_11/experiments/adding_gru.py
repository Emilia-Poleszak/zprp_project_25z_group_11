import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from zprp_project_25z_group_11.generators.components import Components

class AddingGRU(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)

def train_gru(max_steps=300_000, batch_size=20, seq_len=100, hidden_size=128):
    model = AddingGRU(hidden_size=hidden_size)
    model.train()

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=0.001,
        alpha=0.9
    )

    criterion = nn.MSELoss()

    generator = Components(min_no_samples=seq_len, value_range=(-1.0, 1.0))

    recent_errors = []
    recent_correct = []
    window = 2000

    for step in range(1, max_steps + 1):

        seqs = []
        targets = []

        for _ in range(batch_size):
            seq, target = generator.generate()
            seqs.append(seq)
            targets.append(target)

        x = torch.tensor(seqs, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        abs_error = torch.abs(output - y).view(-1)
        correct = (abs_error < 0.04).float()

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
                f"Step {step:6d} | "
                f"Loss {loss.item():.4f} | "
                f"AvgErr {avg:.4f} | "
                f"Correct {int(corr)}/{n}"
            )

            evaluate_model(model, seq_len=seq_len)

        if len(recent_correct) == window and sum(recent_correct) == window:
            print(f"\n*** SUCCESS at step {step}! ***")
            break

    return model

@torch.no_grad()
def evaluate_model(model, num_samples=2000, seq_len=100, threshold=0.04):
    model.eval()

    generator = Components(
        min_no_samples=seq_len,
        value_range=(-1.0, 1.0)
    )

    errors = []
    correct = 0

    for _ in range(num_samples):
        seq, target = generator.generate()

        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([[target]], dtype=torch.float32)

        output = model(x)
        err = torch.abs(output - y).item()

        errors.append(err)
        if err < threshold:
            correct += 1

    avg_error = sum(errors) / len(errors)
    accuracy = correct / num_samples

    print("\n=== Evaluation ===")
    print(f"Avg error: {avg_error:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    model.train()
    return avg_error, accuracy


if __name__ == "__main__":
    trained_model = train_gru()
    evaluate_model(trained_model)