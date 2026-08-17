import sys
import argparse
from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import COLORS_HOUSES
from utils.parsing import parse_hist_args


def histogram(data: pd.DataFrame) -> None:
    """Plot histograms for each numeric column in the dataset,
    grouped by Hogwarts House."""
    numeric_col = [
        col for col in data.columns
        if pd.api.types.is_float_dtype(data[col])
    ]
    nb_col = len(numeric_col)
    graph_per_row = 5
    nb_row = nb_col // graph_per_row + (1 if nb_col % graph_per_row else 0)
    fig = plt.figure(figsize=(6 * graph_per_row, 5 * nb_row), num="Histogram")
    fig.suptitle("Histograms by Hogwarts House for each feature",
                 size=25, y=0.98)

    for i, col in enumerate(numeric_col):
        plt.subplot(nb_row, graph_per_row, i + 1)
        for house, color in COLORS_HOUSES.items():
            color_w_transparency = (*color, 0.5)
            plt.hist(
                x=data[data["Hogwarts House"] == house][col],
                fc=color_w_transparency,
                label=house
            )
            plt.xlabel("Mark", fontsize=16)
            plt.ylabel("Number of students", fontsize=16)
        plt.title(col, fontsize=20)

    fig.legend(
        handles=[Patch(color=color, label=house)
                 for house, color in COLORS_HOUSES.items()],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(COLORS_HOUSES),
        frameon=False,
        fontsize=12
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def main():
    try:
        args: argparse.Namespace = parse_hist_args()
    except Exception as e:
        print(f"Unexpected error: parse_hist_args(): {e}")
        sys.exit(1)

    try:
        histogram(args.dataset)
    except Exception as e:
        print(f"Unexpected error: histogram(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
