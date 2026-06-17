import csv
import pandas as pd
from pathlib import Path


def find_separator(file: Path) -> str:
    """Find the separator used in a CSV file."""
    sniffer = csv.Sniffer()
    with open(file) as f:
        separator: str = sniffer.sniff(f.read(5000)).delimiter

    return separator


def get_data(path: Path) -> pd.DataFrame:
    """Read a CSV file and return a pandas DataFrame."""
    separator: str = find_separator(path)
    data: pd.DataFrame = pd.read_csv(path, sep=separator)
    if "Index" in data.columns:
        data = data.set_index("Index")
    return data
