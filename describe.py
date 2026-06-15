import numpy as np
import pandas as pd
import Typing as tp


DATA_PATH = "datasets/dataset_train.csv"


def get_data(path: str) -> pd.DataFrame:
    # TODO: Eventually add separator detection
    data: pd.DataFrame = pd.read_csv(path, sep=",")
    data.set_index("Index")
    return data


def ft_count(data: pd.Series[float]) -> int:
    count: int = 0
    for _ in data:
        count += 1
    return count


def ft_mean(data: pd.Series[float], count: int) -> float:
    # TODO: Calculate me
    mean = 0
    return mean


def ft_boundaries(data: pd.Series[float], count: int):
    # TODO: Calculate me
    min, max = 0, 0
    return min, max


def ft_std(data: pd.Series[float], mean: float, count: int):
    # TODO Somme ecarts moyenne / n
    std: float = 0
    return std


class Quartiles(tp.NamedTuple):
    quartile_25: float
    quartile_50: float
    quartile_75: float


def ft_quartiles(data: pd.Series[float]) -> dict:
    quartiles = Quartiles(quartile_25=25, quartile_50=23, quartile_75=50)
    # TODO: Calculate me
    return quartiles

def describe(data: pd.DataFrame):
    numeric_col = [col for col in data.columns if data[col].dtype == float]
    statistics = []
    for col in numeric_col:
        col_data: pd.Series[float] = data[col]
        count = ft_count(col_data)
        mean = ft_mean(col_data, count)
        min, max = ft_boundaries(col_data, count)
        quartiles = ft_quartiles(col_data)
        statistics.append({
            "Count": count,
            "Mean": mean,
            "Std": ft_std(col_data, mean, count),
            "Min": min,
            "25%": quartiles[0],
            "50%": quartiles[1],
            "75%": quartiles[2],
            "Max": max
        })
        # TODO Supp: Variance ? mediane, amplitude



def main():
    # TODO: Add ArgParser to fill path + Parsing
    data: pd.DataFrame = get_data(DATA_PATH)
    print(data.head)


if __name__ == "__main__":
    main()