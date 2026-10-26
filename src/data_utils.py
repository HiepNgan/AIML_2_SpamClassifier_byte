"""Load and clean the SMS Spam Collection dataset.

Shared by explore.py, train.py and predict.py so every step
sees the data exactly the same way.
"""
import csv
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "spam.csv"
OUTPUT_DIR = ROOT / "outputs"

LABEL_MAP = {"ham": 0, "spam": 1}

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_PHONE = re.compile(r"\b\d{5,}\b")      # long digit runs = phone / short codes
_MONEY = re.compile(r"[£$€]\s?\d+")
_NUMBER = re.compile(r"\d+")


def clean_text(text: str) -> str:
    """Normalize one message.

    Spam is full of links, phone numbers and prize amounts. The exact
    value does not matter, only that it is there, so each one becomes
    a single token the model can learn from.
    """
    text = text.lower()
    text = _URL.sub(" urltoken ", text)
    text = _MONEY.sub(" moneytoken ", text)
    text = _PHONE.sub(" phonetoken ", text)
    text = _NUMBER.sub(" numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


UCI_PATH = ROOT / "data" / "SMSSpamCollection"


def _read_raw(path: Path) -> pd.DataFrame:
    """Support both versions of the same dataset."""
    if path.exists():
        # Kaggle CSV: not UTF-8, plus 3 empty junk columns after v1, v2.
        raw = pd.read_csv(path, encoding="latin-1")
        return raw.iloc[:, :2]

    if UCI_PATH.exists():
        # UCI original: "label<TAB>message", no header. Messages contain
        # quote characters, so quoting must be switched off.
        return pd.read_csv(UCI_PATH, sep="\t", header=None, quoting=csv.QUOTE_NONE,
                           encoding="utf-8", encoding_errors="replace")

    raise FileNotFoundError(
        f"Không tìm thấy dataset trong {path.parent}.\n"
        "Cần data/spam.csv (Kaggle) hoặc data/SMSSpamCollection (UCI)."
    )


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw dataset and return a tidy frame.

    Columns returned: label (ham/spam), text, target (0/1).
    """
    df = _read_raw(path).copy()
    df.columns = ["label", "text"]

    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["label"].isin(LABEL_MAP) & (df["text"] != "")]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    df["target"] = df["label"].map(LABEL_MAP)
    return df


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple stats used for exploration plots."""
    df = df.copy()
    df["char_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    return df
