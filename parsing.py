import argparse
import pandas as pd
from pathlib import Path


class ValidateCsv(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values.is_file():
            parser.error(f"The file {values} does not exist.")
        try:
            data = self.get_data(values)
            setattr(namespace, self.dest, data)
        except Exception:
            parser.error(f"Unable to read {values} as a pd.DataFrame")

    def get_data(self, path: Path) -> pd.DataFrame:
        # TODO: Eventually add separator detection
        data: pd.DataFrame = pd.read_csv(path, sep=",")
        if "Index" in data.columns:
            data = data.set_index("Index")
        return data


def parse_describe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to visualise dataset's basic statistic"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        action=ValidateCsv,
        help="The dataset to be visualised"
    )

    return parser.parse_args()


def parse_pair_plot_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to visualise dataset's basic statistic"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        action=ValidateCsv,
        help="The dataset to be visualised"
    )

    return parser.parse_args()

def parse_hist_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to visualise dataset's basic statistic"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        action=ValidateCsv,
        help="The dataset to be visualised"
    )

    return parser.parse_args()