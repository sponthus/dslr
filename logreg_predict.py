from __future__ import annotations
import sys
import argparse
from parsing import parse_predictor_args
from logreg import Logreg


# TODO: Check if every exception is correctly catched
def main():
    try:
        args: argparse.Namespace = parse_predictor_args()
    except Exception as e:
        print(f"Unexpected error: parse_describe_args(): {e}")
        sys.exit(1)

    try:
        test: Logreg = Logreg.from_file(args.model)
        test.predictor(args.dataset, drop_na=args.drop_na)

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
    except AssertionError as ae:
        print(ae)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
