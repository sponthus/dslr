import sys
import pandas as pd
from parsing import parse_describe_args
import argparse
from utils import Percentiles, ft_percentiles, ft_count, ft_mean, ft_deviation, ft_trimean


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
        trimean: float = ft_trimean(percentiles)
        variance, std = ft_deviation(col_data, mean, count)
        statistics.append({
            "Name": col,
            "Count": count,
            "Mean": mean,
            "Trimean": trimean,
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
