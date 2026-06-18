import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from parsing import parse_hist_args


COLORS_HOUSES = {
    "Slytherin": (0, 1, 0),
    "Gryffindor": (1, 0, 0),
    "Ravenclaw": (0, 0, 1),
    "Hufflepuff": (1, 1, 0)
}


def histogram(data: pd.DataFrame) -> None:
    """Plot histograms for each numeric column in the dataset,
    grouped by Hogwarts House."""
    numeric_col = [col for col in data.columns if data[col].dtype == float]

    for col in numeric_col:
        plt.figure(num="Histogram")
        for house, color in COLORS_HOUSES.items():
            color_w_transparency = (*color, 0.5)
            plt.hist(
                x=data[data["Hogwarts House"] == house][col],
                fc=color_w_transparency,
                label=house
            )
            plt.xlabel("Mark")
            plt.ylabel("Number of students")
            plt.legend()
        plt.title(col)
        plt.show()


def main():
    try:
        args: argparse.Namespace = parse_hist_args()
    except Exception as e:
        print(f"Unexpected error: parse_hist_args(): {e}")
        sys.exit(1)

    try:
        histogram(args.dataset)
    except AssertionError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: histogram(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
