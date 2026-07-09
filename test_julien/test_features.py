import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import json
import statistics
import os
import concurrent.futures
from tqdm import tqdm
from logreg_test import Logreg
from itertools import combinations
import argparse
from utils.parsing import parse_logreg_train_args

chosen_cols = [
    "Astronomy",
    "Herbology",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Charms"
    ]

def make_all_combinations(chosen_cols):
    all_combinations = []
    for r in range(2, len(chosen_cols) + 1):
        comb = combinations(chosen_cols, r)
        all_combinations.extend(comb)
    return all_combinations


def _logreg_train_from_path(dataset_path: str, chosen_cols: list) -> float:
    # load dataset inside worker process to avoid pickling big DataFrame
    from utils.utils import get_data
    data = get_data(Path(dataset_path))
    # create a fresh Logreg instance in the worker
    from logreg_test import Logreg
    test = Logreg(verbose=False)
    class_col = "Hogwarts House"
    return test.trainer(
        data, nb_cycles=500,
        learning_rate=0.01,
        class_col=class_col,
        features_cols=chosen_cols,
        batch_size=0
    )

def logreg_train(
        data: pd.DataFrame,
        chosen_cols: list
        ) -> float:
    class_col = "Hogwarts House"
    test = Logreg(verbose=False)
    return test.trainer(
        data, nb_cycles=400,
        learning_rate=0.005,
        class_col=class_col,
        features_cols=chosen_cols,
        batch_size=0
        )

def save_results_json(results: dict) -> None:
    serializable_results = [
        {"features": list(features), "score": score}
        for features, score in results.items()
    ]
    with open("results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

def main():
    all_combinations = make_all_combinations(chosen_cols)
    print(f"Number of combinations: {len(all_combinations)}")

    results: dict = {}

    # number of repeated runs per combination
    num_runs = 20
    # configurable worker count via environment variable; default to min(num_runs, cpu_count)
    default_workers = min(num_runs, (os.cpu_count() or 4))
    num_workers = int(os.getenv("NUM_WORKERS", str(default_workers)))

    print(f"Running {num_runs} runs per combination with {num_workers} workers...")


    def _find_csv_path(argv):
        for a in argv[1:]:
            p = Path(a)
            if p.suffix == ".csv" and p.exists():
                return str(p)
        return None

    dataset_path = _find_csv_path(sys.argv)

    if dataset_path:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            for _, comb in tqdm(enumerate(all_combinations), total=len(all_combinations)):
                futures = [ex.submit(_logreg_train_from_path, dataset_path, list(comb)) for _ in range(num_runs)]
                scores: list[float] = [f.result() for f in concurrent.futures.as_completed(futures)]
                results[comb] = statistics.mean(scores)

        save_results_json(results)

if __name__ == "__main__":
    main()