import sys
import pandas as pd
from pathlib import Path

from tqdm.asyncio import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logreg_test import Logreg
from utils.utils import get_data

def calculate_trimean(series: pd.Series) -> float:
    """Calculate the trimean of a pandas Series, excluding NaN values."""

    # Drop NaN values from the series
    clean_series = series.dropna()

    # Calculate the count of non-NaN values
    count = len(clean_series)

    if count == 0:
        raise ValueError("Series contains only NaN values.")

    # Sort the series to calculate percentiles
    sorted_series = clean_series.sort_values()

    # Calculate quartiles
    q1 = sorted_series.quantile(0.25)
    q2 = sorted_series.quantile(0.50)  # This is the median
    q3 = sorted_series.quantile(0.75)

    # Calculate trimean
    trimean = (q1 + 2 * q2 + q3) / 4

    return trimean


def replace_nan_with_trimean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN values in each column of the DataFrame with the trimean of that column."""
    df_trimean = df.copy()  # Create a copy to avoid modifying the original
    for column in df_trimean.columns:
        if df_trimean.dtypes[column] not in [float, int]:
            continue  # Skip non-numeric columns

        trimean = calculate_trimean(df_trimean[column])
        df_trimean[column] = df_trimean[column].fillna(trimean)

    return df_trimean


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


def print_nan_cells_info(df: pd.DataFrame):
    total_cells = df.size
    nan_cells = df.isna().sum().sum()
    nan_percentage = (nan_cells / total_cells) * 100

    print(f"Total cells: {total_cells}")
    print(f"NaN cells: {nan_cells}")
    print(f"Percentage of NaN cells: {nan_percentage:.2f}%")

def print_nan_columns_info(df: pd.DataFrame):
    nan_columns = df.columns[df.isna().any()].tolist()
    print(f"Columns with NaN values: {nan_columns}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_trimean.py <data_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    df = get_data(data_file)

    df_trimean = replace_nan_with_trimean(df)

    nb_runs = 50
    # chosen_cols = [
    #     'Ancient Runes',
    #     'Charms',
    #     'Divination',
    #     'Herbology',
    #     'History of Magic',
    #     'Muggle Studies',
    #     'Transfiguration'
    # ]

    # chosen_cols = [
    #     "Astronomy",
    #     "Divination",
    #     "Herbology"
    #     "History of Magic",
    #     "Muggle Studies",
    #     "Transfiguration",
    # ]

    chosen_cols = [
        "Astronomy",
        'Charms',
        'History of Magic',
        'Muggle Studies',
        'Transfiguration'
    ]

    print(f"Chosen columns: {chosen_cols}")

    scores = []

    print(f"Training without trimean imputation for {nb_runs} runs...")
    for _ in tqdm(range(nb_runs)):
        score, len_data, len_training, len_validator = logreg_train(df, chosen_cols)
        scores.append(score)
    
    mean_score = sum(scores) / len(scores)


    trimean_scores = []
    print(f"Training with trimean imputation for {nb_runs} runs...")
    for _ in tqdm(range(nb_runs)):
        score, len_trimeandata, len_trimeantraining, len_trimeanvalidator = logreg_train(df_trimean, chosen_cols)
        trimean_scores.append(score)
    
    trimean_mean_score = sum(trimean_scores) / len(trimean_scores)

    print(f"Data size: {len_data}")
    print(f"Data:")
    print(f"\tSize of training data: {len_training}")
    print(f"\tSize of validator data: {len_validator}")
    print(f"\tMean score: {mean_score:.4f}")
    print(f"\nData with trimean imputation:")
    print(f"\tSize of training data: {len_trimeantraining}")
    print(f"\tSize of validator data: {len_trimeanvalidator}")
    print(f"\tMean score: {trimean_mean_score:.4f}")

if __name__ == "__main__":
    main()