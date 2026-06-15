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
    mean: float = float("NaN")
    total: int = 0

    for item in data:
        total += item

    mean = total / count

    return mean


def ft_deviation(data: pd.Series[float], mean: float, count: int) -> tuple:
    variance: float = (
        sum((x - mean) ** 2 for x in data)
        / count
    )
    std: float = variance ** 0.5
    return variance, std


class Percentiles(tp.NamedTuple):
    min: float
    quartile_25: float
    quartile_50: float
    quartile_75: float
    max: float


def ft_percentiles(data: pd.Series[float], count: int) -> Percentiles:
    sorted_data: pd.Series[float] = data.sort()

    min: float = sorted_data[0]
    max: float = sorted_data[-1]

    if count % 2 == 0:
        quartile_25=float(sorted_data[round(count * 1 / 4) - 1])
        quartile_50=float(sorted_data[round(count * 1 / 2) - 1])
        quartile_75=float(sorted_data[round(count * 3 / 4) - 1])
    else:
        quartile_25=float(sorted_data[round((count + 1) * 1 / 4) - 1])
        quartile_50=float(sorted_data[round((count + 1) * 1 / 4) - 1])
        quartile_75=float(sorted_data[round((count + 1) * 1 / 4) - 1])

    percentiles = Percentiles(
        min=min,
        quartile_25=quartile_25, 
        quartile_50=quartile_50, 
        quartile_75=quartile_75,
        max=max
    )
    return percentiles

def describe(data: pd.DataFrame):
    numeric_col = [col for col in data.columns if data[col].dtype == float]
    statistics = []
    for col in numeric_col:
        col_data: pd.Series[float] = data[col].dropna()
        count = ft_count(col_data)
        mean = ft_mean(col_data, count)
        percentiles: Percentiles = ft_percentiles(col_data)
        variance, std = ft_deviation(data, mean, count)
        statistics.append({
            "Count": count,
            "Mean": mean,
            "Std": std,
            "Min": percentiles.min,
            "25%": percentiles.quartile_25,
            "50%": percentiles.quartile_50,
            "75%": percentiles.quartile_75,
            "Max": percentiles.max
        })
        # TODO Supp: Variance ? amplitude



def main():
    # TODO: Add ArgParser to fill path + Parsing
    data: pd.DataFrame = get_data(DATA_PATH)
    print(data.head)


if __name__ == "__main__":
    main()