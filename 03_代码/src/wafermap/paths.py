from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "02_数据"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

EXPERIMENT_DIR = PROJECT_ROOT / "04_实验"
RESULT_DIR = PROJECT_ROOT / "05_结果"

if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Raw data directory exists: {RAW_DATA_DIR.exists()}")