import argparse
import pandas as pd
from pathlib import Path


class ValidateCsv(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values.is_file():
            parser.error(f"The file {values} does not exist.")

        data = pd.read_csv(values)
        setattr(namespace, self.dest, values)


def parse_describe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to visualise dataset's basic statistic"
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        action=ValidateCsv,
        help='The dataset to be visualised'
    )

    return parser.parse_args()
