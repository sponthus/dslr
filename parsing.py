import argparse
from pathlib import Path
from utils import get_data
import typing as tp


class ValidateCsv(argparse.Action):
    def __call__(self,
                 parser: argparse.ArgumentParser,
                 namespace: argparse.Namespace,
                 values: str | tp.Sequence[tp.Any] | None,
                 option_string: str | None = None):

        path = tp.cast(Path, values)
        if not path.is_file():
            parser.error(f"The file {path} does not exist.")

        if path.suffix != ".csv":
            parser.error(f"ValueError: The file '{path}' is not a .csv file")

        try:
            data = get_data(path)
            setattr(namespace, self.dest, data)
        except Exception:
            parser.error(f"Unable to read {path} as a pd.DataFrame")


def add_csv_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        type=Path,
        action=ValidateCsv,
        help="The dataset to be visualised"
    )


def parse_describe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to visualise dataset's basic statistic"
    )
    add_csv_dataset_argument(parser)
    return parser.parse_args()


def parse_pair_plot_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to visualise dataset's chosen features \
            in a paired-plot"
    )
    add_csv_dataset_argument(parser)
    return parser.parse_args()


def parse_scatter_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to visualise dataset's similar features \
            in a scatterplot"
    )
    add_csv_dataset_argument(parser)

    return parser.parse_args()


def parse_hist_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to visualise dataset's features \
            in a histogram"
    )
    add_csv_dataset_argument(parser)
    return parser.parse_args()


def parse_logreg_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to train a model with "
                    "logistic regression method"
    )
    add_csv_dataset_argument(parser)
    return parser.parse_args()

def parse_predictor_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to train a model with "
                    "logistic regression method"
    )
    add_csv_dataset_argument(parser)

    parser.add_argument(
        "model",
        type=Path,
        help="The train model used for the prediction"
    )

    return parser.parse_args()