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
LOGS_DIR = REPORTS_DIR / "logs"
MULTIPLICATION_LOGS_DIR = LOGS_DIR / "multiplication"
ADDING_LOGS_DIR = LOGS_DIR / "adding"

# Files
RAW_REBER_DATA_FILENAME = 'reber.txt'

# Datasets parameters
REBER_SAMPLES = 10000
MULTIPLICATION_SEQUENCES = 2560

# Multiplication experiment parameters
MULTIPLICATION_LEARNING_RATE = 0.001
MULTIPLICATION_HIDDEN_SIZE = 64
MULTIPLICATION_SEQUENCE_LENGTH = 100

# Adding experiment parameters
ADDING_LR = 1e-3
ADDING_HIDDEN_SIZE = 64

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
