import sys
import numpy as np
import pandas as pd
import typing as tp
from parsing import parse_describe_args
import argparse
from utils import ft_count, ft_mean, ft_deviation


class Percentiles(tp.NamedTuple):
    """A named tuple to hold the percentiles of a dataset"""
    min: float
    quartile_25: float
    quartile_50: float
    quartile_75: float
    max: float


def ft_percentiles(data: pd.Series, count: int) -> Percentiles:
    """Calculate the percentiles of a pandas Series"""
    assert count != 0, "count can not be null"
    sorted_data: np.ndarray = np.sort(data)

    min: float = sorted_data[0]
    max: float = sorted_data[-1]

    if count % 2 == 0:
        quartile_25 = float(sorted_data[round(count * 1 / 4) - 1])
        quartile_50 = float(sorted_data[round(count * 1 / 2) - 1])
        quartile_75 = float(sorted_data[round(count * 3 / 4) - 1])
    else:
        quartile_25 = float(sorted_data[round((count + 1) * 1 / 4) - 1])
        quartile_50 = float(sorted_data[round((count + 1) * 1 / 4) - 1])
        quartile_75 = float(sorted_data[round((count + 1) * 1 / 4) - 1])

    percentiles = Percentiles(
        min=min,
        quartile_25=quartile_25,
        quartile_50=quartile_50,
        quartile_75=quartile_75,
        max=max
    )
    return percentiles


def describe(data: pd.DataFrame):
    """Describe the basic statistics of a pandas DataFrame"""
    numeric_col = [col for col in data.columns if data[col].dtype == float]
    statistics = []
    for col in numeric_col:
        col_data: pd.Series[float] = data[col].dropna()
        count = ft_count(col_data)
        # Column full of NaN
        if count == 0:
            continue
        mean = ft_mean(col_data, count)
        percentiles: Percentiles = ft_percentiles(col_data, count)
        variance, std = ft_deviation(col_data, mean, count)
        statistics.append({
            "Name": col,
            "Count": count,
            "Mean": mean,
            "Std": std,
            "Variance": variance,
            "Min": percentiles.min,
            "25%": percentiles.quartile_25,
            "50%": percentiles.quartile_50,
            "75%": percentiles.quartile_75,
            "Max": percentiles.max,
            "Amplitude": percentiles.max - percentiles.min
        })
    statistics_df = pd.DataFrame(statistics)
    statistics_df = statistics_df.set_index("Name")
    statistics_df = statistics_df.T
    print(statistics_df)


def main():
    try:
        args: argparse.Namespace = parse_describe_args()
    except Exception as e:
        print(f"Unexpected error: parse_describe_args(): {e}")
        sys.exit(1)

    try:
        describe(args.dataset)
    except AssertionError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: describe(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
