from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Files
RAW_REBER_DATA_FILENAME = 'reber.txt'
ADDING_DATA_FILENAME = 'adding.txt'
MULTIPLICATION_DATA_FILENAME = 'multiplication.txt'

# Datasets parameters
REBER_SAMPLES = 256 # This value is mentioned in original Hochreiter experiment
ADDING_SEQUENCES = 2560
MULTIPLICATION_SEQUENCES = 2560

# Experiment setup parameters
TRAIN_TEST_SPLIT = 0.8

# Utils

REBER_ALPHABET = {c: i for i, c in enumerate('BTSXPVE')}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
