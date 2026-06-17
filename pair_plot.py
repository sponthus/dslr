import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from parsing import parse_pair_plot_args


COLORS_HOUSES = {
    "Slytherin": (0, 1, 0),
    "Gryffindor": (1, 0, 0),
    "Ravenclaw": (0, 0, 1),
    "Hufflepuff": (1, 1, 0)
}


# chosen_cols = [
#     "Astronomy",
#     "Herbology",
#     "Divination",
#     "Muggle Studies",
#     "Ancient Runes",
#     "History of Magic",
#      "Transfiguration",
#     "Charms"
#     ]
# Red = "History of Magic", "Transfiguration"
# chosen_cols = ["Muggle Studies",
#                "History of Magic", "Transfiguration",
#                "Divination",
#                "Astronomy", "Herbology"
#                ]

def pair_plot(data: pd.DataFrame) -> None:
    numeric_col = [col for col in data.columns if data[col].dtype == float]
    chosen_cols = numeric_col
    nb = len(chosen_cols)
    plt.figure(figsize=(nb * 3, nb * 2))

    i = 1
    plt.suptitle("Features pair plot", size=50)
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
                plt.ylabel(col_b)
            if b == nb - 1:
                plt.xlabel(col_a)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.legend()
    plt.show()


def main():
    try:
        args: argparse.Namespace = parse_pair_plot_args()
    except Exception as e:
        print(f"Unexpected error: parse_describe_args(): {e}")
        sys.exit(1)

    try:
        pair_plot(args.dataset)
    except AssertionError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: describe(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
