import pandas as pd
from pathlib import Path


def get_data(path: Path) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(path, sep=",")
    if "Index" in data.columns:
        data = data.set_index("Index")
    return data
