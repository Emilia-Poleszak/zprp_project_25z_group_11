import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from zprp_project_25z_group_11.generators.components import Components

class AddingLSTM(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.head = nn.Linear(hidden_size, 1)
        self._init_lstm()

    def _init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                h = self.hidden_size
                param.data.zero_()
                param.data[h:2 * h] = 3.0

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        packed_out, _ = self.lstm(packed)

        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        last_out = out[torch.arange(x.size(0)), lengths - 1]

        return self.head(last_out)

def train_lstm(max_steps=300_000, batch_size=8, min_seq_len=100, hidden_size=128):
    model = AddingLSTM(hidden_size=hidden_size)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
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

        if step % 2000 == 0:
            n = min(1000, len(recent_errors))
            avg = sum(recent_errors[-n:]) / n
            corr = sum(recent_correct[-n:])
            print(
                f"Step {step:6d} | "
                f"Loss: {loss.item():.4f} | "
                f"AvgErr: {avg:.4f} | "
                f"Correct: {int(corr)}/{n}"
            )

            evaluate_model(model)

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
    trained_model = train_lstm()

    evaluate_model(trained_model)
