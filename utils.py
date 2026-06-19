import csv
import pandas as pd
from pathlib import Path


def ft_count(data: pd.Series) -> int:
    """Count the number of rows in a pandas Series"""
    count: int = 0
    for _ in data:
        count += 1
    return count


def ft_mean(data: pd.Series, count: int) -> float:
    """Calculate the mean of a pandas Series"""
    assert count != 0, "count can not be null"
    mean: float = float("NaN")
    total: int = 0

    for item in data:
        total += item

    mean = total / count

    return mean


def ft_deviation(data: pd.Series, mean: float, count: int) -> tuple:
    """Calculate the variance and standard deviation of a pandas Series"""
    assert count != 0, "count can not be null"
    variance: float = (
        sum((x - mean) ** 2 for x in data)
        / count
    )
    std: float = variance ** 0.5
    return variance, std


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


def standardise_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the numerical columns of a DataFrame"""
    for column in df.columns:
        if df.dtypes[column] not in [float, int]:
            print(f"Standardisation: column `{column}`\
                  of type `{df.dtypes[column]}` skiped")
            continue

        count: int = ft_count(df[column])
        mean: float = ft_mean(df[column], count)
        _, std = ft_deviation(df[column], mean, count)

        df[column] = (df[column] - mean) / std

    return df
