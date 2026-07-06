from __future__ import annotations
import os
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError, SchemaError
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.utils import standardise_data, print_log
from describe import describe
import matplotlib.pyplot as plt

# To think about: Use of a config file for:
#   - dataset
#   - chosen features
#   - training parameters (cyles, learning rate)


class Logreg():

    def __init__(
            self,
            class_enum: dict[str, int] | None = None,
            trimeans: dict[str, float] | None = None,
            nb_classes: int = 0,
            nb_features: int = 0,
            class_col: str | None = None,
            features_cols: list[str] | None = None,
            weights: np.ndarray | None = None,
            biases: np.ndarray | None = None,
            verbose: bool = False
            ):

        # Remove the saved dataframe?
        self.class_enum: dict | None = class_enum
        self.trimeans: dict | None = trimeans
        self.nb_classes = nb_classes
        self.nb_features = nb_features
        self.class_col: str | None = class_col
        self.features_cols: list[str] | None = features_cols
        self.weights: np.ndarray | None = weights
        self.biases: np.ndarray | None = biases
        self.verbose = verbose

        arguments: list = [
            class_enum,
            trimeans,
            class_col,
            features_cols,
            weights,
            biases
            ]
        # If any argument is not None, check the coherence of given parameters
        if any(arg is not None for arg in arguments):
            self._check_parameters()
        print_log("Logreg class created", self.verbose)

    def _check_parameters(self):
        # Check types
        if not isinstance(self.nb_classes, int):
            raise TypeError("Invalid nb_classes type")
        if not isinstance(self.nb_features, int):
            raise TypeError("Invalid nb_features type")
        if not isinstance(self.features_cols, list):
            raise TypeError("Invalid features_cols type")
        if not isinstance(self.class_col, str):
            raise TypeError("Invalid class_col type")

        if not isinstance(self.class_enum, dict):
            raise TypeError("Invalid class_enum type")
        if not all(isinstance(key, str) for key in self.class_enum.keys()):
            raise TypeError("Invalid class_enum keys type")
        if not all(isinstance(v, int) for v in self.class_enum.values()):
            raise TypeError("Invalid class_enum values type")

        if not all(isinstance(col, str) for col in self.features_cols):
            raise TypeError("Invalid features_cols type")
        if not isinstance(self.weights, np.ndarray):
            raise TypeError("Invalid weights type")
        if not isinstance(self.biases, np.ndarray):
            raise TypeError("Invalid biases type")

        if not isinstance(self.trimeans, dict):
            raise TypeError("Invalid trimeans type")
        if not all(isinstance(key, str) for key in self.trimeans.keys()):
            raise TypeError("Invalid trimeans keys type")
        if not all(isinstance(v, float) for v in self.trimeans.values()):
            raise TypeError("Invalid trimeans values type")

        # Check consistency
        if not len(self.class_enum.keys()) == self.nb_classes:
            raise ValueError(
                "No correspondance between class_enum and nb_classes"
                )
        if not len(self.features_cols) == self.nb_features:
            raise ValueError("Invalid nb_features")
        if not len(self.trimeans.keys()) == self.nb_features:
            raise ValueError("Invalid trimeans")

        # Check shapes
        if not self.weights.shape == (self.nb_features, self.nb_classes):
            raise ValueError("Invalid weights shape")
        if not self.biases.shape == (self.nb_classes, 1):
            raise ValueError("Invalid biases shape")

    @classmethod
    def from_file(cls, verbose: bool, model_path: Path) -> Logreg:
        try:
            with open(model_path, 'r') as f:
                json_str: str = f.read()
            with open("utils/logreg_schema.json", "r", encoding="utf-8") as f:
                schema = json.load(f)
            data: dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise TypeError(f"Wrong json format: {e}")
        except Exception as e:
            raise e

        try:
            validate(instance=data, schema=schema)
            print_log(f"Loaded model {model_path}", verbose)
        except ValidationError as ve:
            raise ValueError(f"Validation error: {ve.message}")
        except SchemaError as se:
            raise ValueError(f"Schema error: {se.message}")
        except Exception as e:
            raise RuntimeError(e)

        model = Logreg(
            data["class_enum"],
            data["trimeans"],
            data["nb_classes"],
            data["nb_features"],
            data["class_col"],
            data["features_cols"],
            np.array(data["weights"]),
            np.array(data["biases"]),
            verbose
        )

        return model

    def __str__(self) -> str:
        base = "Logreg class"
        initialized: bool = self.is_init()
        if not initialized:
            return base + "\nNot initialized"
        base += f"\n{self.class_enum=}"
        base += f"\n{self.trimeans=}"
        base += f"\n{self.nb_classes=}"
        base += f"\n{self.nb_features=}"
        base += f"\n{self.class_col=}"
        base += f"\n{self.features_cols=}"
        base += f"\n{self.weights=}"
        base += f"\n{self.biases=}"
        return base

    # CONDITIONS

    def is_init(self) -> bool:
        """Return True when the model appears initialized."""
        return all([
            self.class_enum is not None,
            self.trimeans is not None,
            self.nb_classes > 0,
            self.nb_features > 0,
            self.class_col is not None,
            self.features_cols is not None,
            self.weights is not None,
            self.biases is not None,
        ])

    def is_compatible(self, data: pd.DataFrame):
        """Checks if a dataset is compatible with the class initialization
        and the class attributes validity."""
        if self.class_enum is None:
            raise ValueError("no enum stored in model")
        if self.trimeans is None:
            raise ValueError("no trimeans stored in model")
        columns = data.columns
        if self.features_cols is None \
           or not all(feature in columns for feature in self.features_cols):
            raise ValueError("not all model features in data")
        for feature in self.features_cols:
            if feature not in self.trimeans.keys():
                raise ValueError(
                    f"{feature} feature not stored in model trimean"
                    )
        if not self.nb_classes == len(self.class_enum):
            raise ValueError("wrong nb_classes stored in model")
        if not self.class_col:
            raise ValueError("no class_col in model")
        if not self.nb_features == len(self.features_cols):
            raise ValueError("wrong nb_features in model")
        if self.weights is None \
           or not self.weights.shape == (self.nb_features, self.nb_classes):
            raise ValueError("wrong weights in model")
        if self.biases is None \
           or not self.biases.shape == (self.nb_classes, 1):
            raise ValueError("wrong biases in model")
        return True

    # INITIALIZATION

    def initialize(
            self,
            data: pd.DataFrame,
            features_cols: list[str],
            class_col: str
            ):
        """Initializes features_cols, class_col, enum,
        weights and biases for the class"""
        self.features_cols = features_cols
        self.class_col = class_col
        classes = data[class_col].unique()
        self.nb_classes = len(classes)
        self.nb_features = len(features_cols)

        # Weights = Initial weights for each feature and each class
        # Or np.random ?
        self.weights = np.full((self.nb_features, self.nb_classes), 0.5)
        # One bias for each class because the biases are factorized in equation
        self.biases = np.zeros((self.nb_classes, 1))
        self.class_enum = {
            name: i for i, name in enumerate(classes)
        }
        statistics_df = describe(data[features_cols])
        self.trimeans = {
            name: statistics_df.loc["Trimean", name] for name in features_cols
        }
        print_log(f"Initialized logreg from dataset: {self}", self.verbose)

    # USAGE

    def _check_batch_size(
            self,
            batch_size: int,
            training_data: pd.DataFrame
            ) -> int:
        if batch_size > len(training_data):
            print(
                "Warning: required batch-size is higher than training "
                "data length, clamped it to training data length "
                "(batch gradient-descent)"
                )
            batch_size = len(training_data)
        elif batch_size == 0:
            batch_size = len(training_data)
        print_log(f"Using {batch_size=}", self.verbose)
        return batch_size

    def _preprocessing(
            self,
            data: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        all_cols = self.features_cols + [self.class_col]
        data = data[all_cols]
        data = data.dropna(axis=0)
        if data.empty:
            raise ValueError("data contains nan only")
        if n != len(data):
            print_log(f"Dropped {n - len(data)} lines with nan",
                      self.verbose)
        data = standardise_data(data, self.verbose)
        data[self.class_col] = data[self.class_col].map(self.class_enum)
        training_data, validator_data = train_test_split(
            data,
            test_size=0.25,
            stratify=data[self.class_col],
            shuffle=True
        )
        print_log(
            f"Split data in training_data (n={len(training_data)}) " +
            f"and validator_data (n={len(validator_data)})",
            self.verbose
            )
        return training_data, validator_data

    def train(
            self,
            data: pd.DataFrame,
            nb_cycles: int,
            learning_rate: float,
            class_col: str,
            features_cols: list[str],
            batch_size: int
            ):
        """
        Initializes and trains a Logreg model from dataset.
        Logreg class must not be already trained to call this function.
        """
        # Check if model already trained
        if self.is_init():
            raise ValueError("model is already initialized")
        else:
            self.initialize(data, features_cols, class_col)

        # TODO: Evaluate if it is better to shuffle the data
        # at each cycle to avoid overfitting
        training_data, validator_data = self._preprocessing(data)
        batch_size = self._check_batch_size(batch_size, training_data)

        # X = features values for each feature and sample
        x = np.array(training_data[self.features_cols])
        # Y = Expected class probability for each sample
        y = np.zeros((self.nb_classes, len(training_data)))
        for i in range(len(training_data)):
            class_index = int(training_data.iloc[i][self.class_col])
            y[class_index][i] = 1

        losses = []
        scores = []
        size = len(training_data)
        factor = size / batch_size
        for cycle in tqdm(range(nb_cycles)):
            shuffled_indices = np.random.permutation(size)
            x_shuffled = x[shuffled_indices]
            y_shuffled = y[:, shuffled_indices]
            epoch_loss = 0
            epoch_score = 0
            for i in range(0, size, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[:, i:i + batch_size]
                logloss, score = self._epoch(x_batch, y_batch, learning_rate)
                epoch_loss += logloss
                epoch_score += score
            epoch_loss /= factor
            epoch_score /= factor
            losses.append(epoch_loss)
            scores.append(epoch_score)

        print_log(f"\nFinished training:\n{self}", self.verbose)
        self.plot(losses, name="Losses through training")
        self.plot(scores, name="Accuracy scores through training")

        y_validator = np.zeros((self.nb_classes, len(validator_data)))
        for i in range(len(validator_data)):
            class_index = int(validator_data.iloc[i][class_col])
            y_validator[class_index][i] = 1
        y_validator = np.argmax(y_validator, axis=0)
        validator_data = np.array(validator_data[self.features_cols])

        y_pred_validator = np.argmax(self.predict(validator_data), axis=0)
        score = accuracy_score(y_true=y_validator, y_pred=y_pred_validator)
        print_log(f"Accuracy score={score*100:.3f}%", self.verbose)

    def _epoch(self, x: np.ndarray, y: np.ndarray, learning_rate: float):
        y_pred: np.ndarray = self.predict(x)
        gradient_w, gradient_b = self.compute_gradient(x, y_pred, y)

        # For graphical representations
        logloss = self.log_loss(y, y_pred)
        score = accuracy_score(
            y_true=np.argmax(y, axis=0),
            y_pred=np.argmax(y_pred, axis=0)
            )

        self.weights = self.update(self.weights, gradient_w, learning_rate)
        self.biases = self.update(self.biases, gradient_b, learning_rate)
        return logloss, score

    def predictor(self, data: pd.DataFrame, drop_na: bool = True) -> None:
        """Used to predict values from a trained model"""
        if not self.is_init():
            raise ValueError("model is not initialized")

        if not self.is_compatible(data):
            raise ValueError("model training is not compatible with data")

        columns = ["Index"]
        columns.extend(self.features_cols)

        data.reset_index(inplace=True)
        data = data[columns]
        data = data.set_index("Index")
        n = len(data)
        if drop_na:
            data = data.dropna(axis=0)
            if n != len(data):
                print_log(f"Dropped {n - len(data)} lines with nan",
                          self.verbose)
        else:
            data = self._replace_na(data)
            print_log("Replaced nan with known trimeans", self.verbose)
        data = standardise_data(data)

        x: np.ndarray = np.array(data)
        y_pred = self.predict(x)

        enum_by_id = {
            value: key for key, value in self.class_enum.items()
        }

        str_results = []
        class_index = np.argmax(y_pred.T, axis=1)
        for i in range(len(x)):
            str_results.append(enum_by_id.get(class_index[i]))

        data[self.class_col] = str_results
        data.to_csv(
            "houses.csv",
            sep=",",
            index_label="Index",
            columns=[self.class_col]
            )
        print_log(f"Results:\n{data[self.class_col]}", self.verbose)
        print_log("Results saved in houses.csv", True)

    # COMPUTATIONS
    # TODO: Move every stats calculation from class into different file?

    def update(
            self,
            to_update: np.ndarray,
            gradient: np.ndarray,
            learning_rate: float
            ) -> np.ndarray:
        """Updates weights or bias with gradient modulated by learning_rate"""
        return to_update - (gradient * learning_rate)

    def predict(self, x: np.ndarray) -> np.ndarray:
        raw_result = self.weights.T @ x.T + self.biases
        y_pred: np.ndarray = self.sigmoid(raw_result)
        return y_pred

    def log_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Loss function or log loss, for visualization"""
        res: float = -(y * np.log(y_pred)
                       + (1 - y) * np.log(1 - y_pred)).mean()
        return res

    def compute_gradient(
            self,
            x: np.ndarray,
            y_pred: np.ndarray,
            y: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
        """Uses derivative from log loss function, for gradient descent"""
        error = y_pred - y
        gradient_w = (x.T @ error.T) / x.shape[0]
        error_b = y_pred - y
        gradient_b = np.sum(error_b, axis=1, keepdims=True) / x.shape[0]
        return gradient_w, gradient_b

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function, turns any value to 0-1"""
        res: np.ndarray = 1 / (1 + np.exp(-x))
        return res

    def _replace_na(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replaces nan values with known trimean for the feature

        If all features have no data, the row is deleted"""
        df_copy: pd.DataFrame = data.copy()
        for index, row in data.iterrows():
            missing = 0
            for feature in self.features_cols:
                if np.isnan(row[feature]):
                    missing += 1
                    df_copy.loc[index, feature] = self.trimeans.get(feature, 0)
            if missing == self.nb_features:
                df_copy = df_copy.drop(index=index)
        return df_copy

    # LOAD AND SAVE

    def save_weights(self):
        """Save weights to a file"""
        # Use json file
        # Save used feature for training and enum

        save_dict = {
            "class_enum": self.class_enum,
            "nb_classes": self.nb_classes,
            "nb_features": self.nb_features,
            "class_col": self.class_col,
            "trimeans": self.trimeans,
            "features_cols": self.features_cols,
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
        }

        model_folder: str = "models"
        name_format: str = "DSLR_model_%Y-%m-%d_%H-%M-%S.json"
        model_file: str = datetime.now().strftime(name_format)
        model_path = os.path.join(model_folder, model_file)

        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        with open(model_path, 'w') as f:
            json.dump(save_dict, f, indent=4)

    # TRACK
    # TODO: Add more figure for training stats visualisation (accuracy)

    def plot(self, data: list, name: str):
        plt.figure()
        plt.title(name)
        plt.plot(data)
        plt.show()
