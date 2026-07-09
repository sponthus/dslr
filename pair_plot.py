import sys
import argparse
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.utils import COLORS_HOUSES
from utils.parsing import parse_pair_plot_args


def pair_plot(data: pd.DataFrame) -> None:
    """Plot pair plots for each numeric column in the dataset,
    grouped by Hogwarts House."""

    def wrap_label(label: str, width: int = 12) -> str:
        return textwrap.fill(label, width=width,
                             break_long_words=False,
                             break_on_hyphens=False)

    numeric_col = [
        col for col in data.columns
        if pd.api.types.is_float_dtype(data[col])
    ]
    chosen_cols = numeric_col
    nb = len(chosen_cols)
    fig = plt.figure(figsize=(nb * 2.5, nb * 1.5), num="Pair Plot")

    i = 1
    fig.suptitle("Features pair plot", size=25, y=0.98)
    for b, col_b in enumerate(chosen_cols):
        for a, col_a in enumerate(chosen_cols):
            plt.subplot(nb, nb, i)
            i += 1

            if col_a == col_b:
                for house, color in COLORS_HOUSES.items():
                    color_w_transparency = (*color, 0.5)
                    plt.hist(
                        label=house,
                        x=data[data["Hogwarts House"] == house][col_a],
                        fc=color_w_transparency
                    )

            else:
                for house, color in COLORS_HOUSES.items():
                    color_w_transparency = *color, 0.5
                    filtered_data = data[data["Hogwarts House"] == house]
                    plt.scatter(
                        label=house,
                        x=filtered_data[col_a],
                        y=filtered_data[col_b],
                        fc=color_w_transparency,
                        s=4
                    )
            if a == 0:
                plt.ylabel(wrap_label(col_b), rotation=45, labelpad=20)
            if b == nb - 1:
                plt.xlabel(col_a)

    fig.legend(
        handles=[Patch(color=color, label=house)
                 for house, color in COLORS_HOUSES.items()],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(COLORS_HOUSES),
        frameon=False,
        fontsize=12
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def main():
    try:
        args: argparse.Namespace = parse_pair_plot_args()
    except Exception as e:
        print(f"Unexpected error: parse_pair_plot_args(): {e}")
        sys.exit(1)

    try:
        pair_plot(args.dataset)
    except AssertionError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: pair_plot(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
