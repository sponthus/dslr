from __future__ import annotations
import sys
import pandas as pd
import argparse
from pathlib import Path
from parsing import parse_predictor_args
from logreg_train import LogregTrain


def main():
    try:
        args: argparse.Namespace = parse_predictor_args()
    except Exception as e:
        print(f"Unexpected error: parse_describe_args(): {e}")
        sys.exit(1)

    try:
        test: LogregTrain = LogregTrain.from_file(args.model)
        test.predictor(args.dataset)

    except AssertionError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: describe(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
