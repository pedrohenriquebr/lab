import os
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
MODEL_PATH = MODELS_DIR / "q_table_pc"
BEST_MODEL = "opt_reward_tuning_trial_021"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
COLAB_MODE = False  # 🔴 IMPORTANTE: True para Colab, False para PC local
# Configuração do Robô (AJUSTE SEU IP)
ROBOT_IP = "192.168.1.11"  # 🔴 ALTERE para o IP do seu ESP32
COMMANDS = {
    0: f"http://{ROBOT_IP}/stop",
    1: f"http://{ROBOT_IP}/go",
    2: f"http://{ROBOT_IP}/back",
    3: f"http://{ROBOT_IP}/left",
    4: f"http://{ROBOT_IP}/right"
}
STREAM_URL = f"http://{ROBOT_IP}:81/stream"
DELAY = 0.5  # Segundos entre ações (ajuste conforme necessário)


# --- CONFIGURAÇÃO DO ROBÔ ---
ROBOT_IP = "192.168.1.11"
TIMEOUT = 2

# URLs Originais do ESP32
ESP_CMD_BASE = f"http://{ROBOT_IP}"
ESP_STREAM_URL = f"http://{ROBOT_IP}:81/stream"

# --- CONFIGURAÇÃO DO DATASET ---
DATASET_DIR = "dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
CSV_FILE = os.path.join(DATASET_DIR, "data.csv")

# Mapeamento Ação -> ID (Para treinar a rede depois)
ACTION_MAP = {
    'stop': 0,
    'go': 1,
    'back': 2,
    'left': 3,
    'right': 4
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
