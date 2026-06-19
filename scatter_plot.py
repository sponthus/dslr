import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from parsing import parse_scatter_args


COLORS_HOUSES = {
    "Slytherin": (0, 1, 0),
    "Gryffindor": (1, 0, 0),
    "Ravenclaw": (0, 0, 1),
    "Hufflepuff": (1, 1, 0)
}


def scatter_plot(data: pd.DataFrame, feature_a: str, feature_b: str) -> None:
    """Plot satter_plots to show similar features in the dataset"""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise AssertionError("scatter_plot needs data (pd.DataFrame)")
    if "Hogwarts House" not in data.columns:
        raise AssertionError("data needs `Hogwarts House` column")
    if feature_a not in data.columns:
        raise AssertionError(f"'{feature_a}' not in data")
    if feature_b not in data.columns:
        raise AssertionError(f"'{feature_b}' not in data")

    plt.figure()
    for house, color in COLORS_HOUSES.items():
        color_w_transparency = *color, 0.5
        filtered_data = data[data["Hogwarts House"] == house]
        plt.scatter(
            label=house,
            x=filtered_data[feature_a],
            y=filtered_data[feature_b],
            fc=color_w_transparency,
            s=4
        )
    plt.xlabel(feature_a)
    plt.ylabel(feature_b)
    plt.legend()
    plt.title(f"{feature_a} and {feature_b} similarities")
    plt.show()


def main():
    try:
        args: argparse.Namespace = parse_scatter_args()
    except Exception as e:
        print(f"Unexpected error: parse_pair_plot_args(): {e}")
        sys.exit(1)

    try:
        scatter_plot(args.dataset,
                     feature_a="Astronomy",
                     feature_b="Defense Against the Dark Arts")
    except AssertionError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: pair_plot(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
