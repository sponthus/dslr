import sys
import pandas as pd
import argparse
from parsing import parse_logreg_train_args
from logreg import Logreg


def logreg_train(data: pd.DataFrame, batch_size: int) -> None:
    chosen_cols = [
        "Muggle Studies",
        "History of Magic", "Transfiguration",
        "Divination",
        "Astronomy", "Herbology"
        ]
    class_col = "Hogwarts House"
    # print(data)
    test = Logreg()
    test.train(data, nb_cycles=500, learning_rate=0.00005, class_col=class_col, features_cols=chosen_cols, batch_size=batch_size)

    test.save_weights()

def main():
    try:
        args: argparse.Namespace = parse_logreg_train_args()
    except Exception as e:
        print(f"Unexpected error: parse_describe_args(): {e}")
        sys.exit(1)

    try:
        logreg_train(args.dataset, args.batch_size)
    except AssertionError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: describe(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()