from loguru import logger
from tqdm import tqdm

from zprp_project_25z_group_11.generators.components import Components
from zprp_project_25z_group_11.config import IN_FEATURES, OUT_FEATURES, STATE_FEATURES, LR

import torch
from torch import nn, optim
from LRU_pytorch import LRU

def train_experiment_4(model_path):
    """Trains LRU model for adding problem.

    :param model_path: path to saved model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training LRU model on {device}...")
    model = LRU(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        state_features=STATE_FEATURES
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    generator = Components(min_no_samples=100, value_range=(-1.0, 1.0))
    correct_streak = 0
    total_loss = 0
    for step in tqdm(range(100000)):
        seq, target = generator.generate()
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(1).to(device)  # (T,1,D)
        target = torch.tensor(target, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output[-1], target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = output[-1].item()
        if abs(predicted - target.item()) < 0.5:  # tolerance-based correctness
            correct_streak += 1
        else:
            correct_streak = 0

        avg_loss = total_loss / (step + 1)
        if avg_loss < 0.01 and correct_streak >= 2000:
            logger.success(f"Training converged at step {step}.")
            break

    torch.save(model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}")