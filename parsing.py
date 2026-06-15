import argparse
from pathlib import Path


class ValidateFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values.is_file():
            parser.error(f"The file {values} does not exist.")
        setattr(namespace, self.dest, values)


def parse_describe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A simple program to visualise dataset's basic statistic"
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        action=ValidateFile,
        help='The dataset to be visualised'
    )

    return parser.parse_args()
