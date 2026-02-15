from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / 'data'
LOG_DIR = ROOT_DIR / 'logs'

LOG_DIR.mkdir(exist_ok=True)