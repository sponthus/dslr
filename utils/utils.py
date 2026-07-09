import csv
import pandas as pd
from pathlib import Path
from utils.math import ft_count, ft_mean, ft_deviation

COLORS_HOUSES = {
    "Slytherin": (0, 1, 0),
    "Gryffindor": (1, 0, 0),
    "Ravenclaw": (0, 0, 1),
    "Hufflepuff": (1, 1, 0)
}


def print_log(msg: str, verbose: bool = False) -> None:
    """
    Prints msg on standard output if verbose = True
    """
    if verbose:
        print(msg + "\n")


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


def standardise_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Standardise the numerical columns of a DataFrame"""
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            print_log(
                f"Standardisation: column `{column}` " +
                f"of type `{df.dtypes[column]}` skipped", verbose
            )
            continue

        count: int = ft_count(df[column])
        mean: float = ft_mean(df[column], count)
        _, std = ft_deviation(df[column], mean, count)

        df.loc[:, column] = (df[column] - mean) / std
    print_log("Data standardized", verbose)

    return df
