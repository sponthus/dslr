from __future__ import annotations
import sys
import argparse
import pandas as pd
from pathlib import Path
from utils.parsing import parse_predictor_args
from logreg import Logreg


def logreg_predict(
        data: pd.DataFrame,
        model: Path,
        verbose: bool,
        drop_na: bool
        ) -> None:
    test: Logreg = Logreg.from_file(verbose, model)
    test.predictor(data, drop_na=drop_na)


def main():
    try:
        args: argparse.Namespace = parse_predictor_args()
    except Exception as e:
        print(f"Unexpected error: parse_predictor_args(): {e}")
        sys.exit(1)

    try:
        logreg_predict(
            data=args.dataset,
            model=args.model,
            verbose=args.verbose,
            drop_na=args.drop_na
            )
    except FileNotFoundError as fe:
        print(f"{fe.strerror}: {fe.filename}")
        sys.exit(1)
    except PermissionError as pe:
        print(f"{pe.strerror}: {pe.filename}")
        sys.exit(1)
    except ValueError as ve:
        print(ve)
        sys.exit(1)
    except TypeError as te:
        print(te)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
