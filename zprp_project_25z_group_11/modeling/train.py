from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from zprp_project_25z_group_11.config import MODELS_DIR, PROCESSED_DATA_DIR
from zprp_project_25z_group_11.modeling.exp4_training import train_experiment_4

app = typer.Typer()


@app.command()
def main(
        model_path: Path = MODELS_DIR / "lru_model.pt",
        experiment: int = typer.Option(4, help="Which experiment to run (1, 4, 5)"),
):
    if experiment == 4:
        train_experiment_4(model_path)


if __name__ == "__main__":
    app()
