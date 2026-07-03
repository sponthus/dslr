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
from utils import standardise_data
from describe import describe
import matplotlib.pyplot as plt

# To think about: Use of a config file for:
#   - dataset
#   - chosen features
#   - training parameters (cyles, learning rate)

class Logreg():

    def __init__(
            self,
            enum_by_name: dict[str, int] | None = None,
            trimeans: dict[str, float] | None = None,
            nb_classes: int = 0,
            nb_features: int = 0,
            class_col: str | None = None,
            features_cols: list[str] | None = None,
            weights: np.ndarray | None = None,
            biases: np.ndarray | None = None
            ):

        # Remove the saved dataframe?
        self.enum_by_name: dict | None = enum_by_name
        self.trimeans: dict | None = trimeans
        self.nb_classes = nb_classes
        self.nb_features = nb_features
        self.class_col: str | None = class_col
        self.features_cols: list[str] | None = features_cols
        self.weights: np.ndarray | None = weights
        self.biases: np.ndarray | None = biases

        arguments: list = [enum_by_name, trimeans, class_col, features_cols, weights, biases]
        # If any argument is not None, check the coherence of given parameters
        if any(arg is not None for arg in arguments):
            self._check_parameters()

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

        if not isinstance(self.enum_by_name, dict):
            raise TypeError("Invalid class_enum type")
        if not all(isinstance(key, str) for key in self.enum_by_name.keys()):
            raise TypeError("Invalid enum_by_name keys type")
        if not all(isinstance(v, int) for v in self.enum_by_name.values()):
            raise TypeError("Invalid enum_by_name values type")
        
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
        if not len(self.enum_by_name.keys()) == self.nb_classes:
            raise ValueError("No correspondance between enum_by_name and nb_classes")
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
    def from_file(cls, model_path: Path) -> Logreg:
        try:
            with open(model_path, 'r') as f:
                json_str:str = f.read()
            with open("logreg_schema.json", "r", encoding="utf-8") as f:
                schema = json.load(f)
            data: dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise TypeError(f"Wrong json format: {e}")
        except Exception as e:
            raise e

        try:
            validate(instance=data, schema=schema)
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
            np.array(data["biases"])
        )

        return model

    #### CONDITIONS

    def is_init(self) -> bool:
        if self.enum_by_name is None or self.trimeans is None or self.nb_classes == 0 or self.nb_features == 0 or self.class_col is None or self.features_cols is None or self.weights is None or self.biases is None:
            return False
        return True

    def is_compatible(self, data: pd.DataFrame, training: bool):
        """Checks if a dataset is compatible with the class initialization
        and the class attributes validity."""
        columns = data.columns
        assert self.features_cols is not None and all(feature in columns for feature in self.features_cols), "not all features in data"
        assert self.nb_features == len(self.features_cols), "wrong nb_features"
        assert self.weights is not None and self.weights.shape == (self.nb_features, self.nb_classes), "wrong weights"
        assert self.biases is not None and self.biases.shape == (self.nb_classes, 1), "wrong biases"
        if training:
            assert self.class_col in columns, f"'{self.class_col}' not in data"
            assert self.nb_classes == len(data[self.class_col].unique()), "wrong nb_class"
            assert self.enum_by_name is not None, "no enum stored"
            assert self.trimeans is not None, "no trimeans stored"
            for data_class in data[self.class_col].unique():
                assert self.enum_by_name.get(data_class, False), "Unknown data_class"
                assert self.trimeans.get(data_class, False), "No trimean for data_class"
        
        return True

    #### INITIALIZATION

    def initialize(self, data: pd.DataFrame, features_cols: list[str], class_col: str):
        """Initializes features_cols, class_col, enum, weights and biases for the class"""
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
        self.enum_by_name = {
            name: i for i, name in enumerate(classes)
        }
        statistics_df = describe(data[features_cols])
        self.trimeans = {
            name: statistics_df.loc["Trimean", name] for name in features_cols
        }

    #### USAGE

    def train(
            self,
            data: pd.DataFrame,
            nb_cycles: int,
            learning_rate: float,
            class_col: str,
            features_cols: list[str],
            batch_size: int
            ):

        # Check if model already trained on a specific data
        # + compatibility with known class_col and nb_features
        # Or determine self.class_col / self.features 
        # Store enum if not already present
        if self.is_init():
            if not self.is_compatible(data, training=True):
                raise Exception()
        else:
            self.initialize(
                data=data, 
                features_cols=features_cols, 
                class_col=class_col
            )

        # TODO: Evaluate if it is better to shuffle the data
        # at each cycle to avoid overfitting
        # + factorize as preprocessing()
        all_cols = features_cols + [class_col]
        data = data[all_cols]
        data = data.dropna(axis=0)
        assert not data.empty, "data contains nan only"
        data = standardise_data(data)
        data.loc[:, class_col] = data[class_col].map(self.enum_by_name)
        training_data, validator_data = train_test_split(
            data,
            test_size=0.25,
            stratify=data[class_col],
            shuffle=True
        )
        if batch_size > len(training_data):
            print(f"Warning: required batch-size is higher than training data length, clamped it to training data length (batch gradient-descent)")
            batch_size = len(training_data)
        elif batch_size == 0:
            batch_size = len(training_data)
        # print(f"{training_data=}\n{validator_data=}")

        print(f"{self.features_cols} / {self.class_col}")
        # X = features values for each feature and sample
        x = np.array(training_data[self.features_cols])
        # Y = Expected class probability for each sample
        y = np.zeros((self.nb_classes, len(training_data)))
        for i in range(len(training_data)):
            # print(f"{training_data.iloc[i][self.class_col]=}")
            class_index = int(training_data.iloc[i][self.class_col])
            # print(f"{class_index=}")
            y[class_index][i] = 1
        # print(f"{y=}")

        losses = []
        scores = []
        # print(f"Shape before shuffle: {x.shape=} / {y.shape}")
        size = len(training_data)
        factor = size / batch_size
        for cycle in tqdm(range(nb_cycles)):
            shuffled_indices = np.random.permutation(size)
            # print(f"{shuffled_indices=}")
            x_shuffled, y_shuffled = x[shuffled_indices], y[:, shuffled_indices]
            # print(f"{x_shuffled=}, {y_shuffled=}")
            # print(f"Shape after shuffle: {x_shuffled.shape=} / {y_shuffled.shape}")
            epoch_loss = 0
            epoch_score = 0
            for i in range(0, size, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[:, i:i + batch_size]
                # print(f"{x_batch=} / {y_batch=}")
                logloss, score = self._epoch(x_batch, y_batch, learning_rate)
                epoch_loss += logloss
                epoch_score += score
            epoch_loss /= factor
            epoch_score /= factor
            losses.append(epoch_loss)
            scores.append(epoch_score)
            # print(f"{cycle=} done")
            
            # print(f"{self.weights}")
        self.plot(losses, name="Losses through training")
        # print(f"{scores=}")
        self.plot(scores, name="Accuracy scores through training")

        y_validator = np.zeros((self.nb_classes, len(validator_data)))
        for i in range(len(validator_data)):
            class_index = int(validator_data.iloc[i][class_col])
            y_validator[class_index][i] = 1
        y_validator = np.argmax(y_validator, axis=0)
        validator_data = np.array(validator_data[self.features_cols])
    
        y_pred_validator = np.argmax(self.predict(validator_data), axis=0)
        # print(f"{y_pred_validator=} / {y_validator=}\n")
        score = accuracy_score(y_true=y_validator, y_pred=y_pred_validator)
        print(f"{score=}")

    def _epoch(self, x: np.ndarray, y: np.ndarray, learning_rate: float):
        y_pred: np.ndarray = self.predict(x)
        gradient_w, gradient_b = self.compute_gradient(x, y_pred, y)
        # print(f"{gradient_w=}, \n {gradient_b=}")

        # For graphical representations
        logloss = self.log_loss(y, y_pred)
        score = accuracy_score(y_true=np.argmax(y, axis=0), y_pred=np.argmax(y_pred, axis=0))

        self.weights = self.update(self.weights, gradient_w, learning_rate)
        self.biases = self.update(self.biases, gradient_b, learning_rate)
        return logloss, score

    def predictor(self, data: pd.DataFrame, drop_na: bool = True) -> None:
        """Used to predict values from a trained model"""
        assert self.is_init(), "not initialized"
        assert self.is_compatible(data, training=False), "model training is not compatible with data"

        columns = ["Index"]
        columns.extend(self.features_cols)

        data.reset_index(inplace=True)
        data = data[columns]
        data = data.set_index("Index")
        if drop_na:
            data = data.dropna(axis=0)
        else:
            data = self._replace_na(data)
        data = standardise_data(data)

        x: np.ndarray = np.array(data)
        y_pred = self.predict(x)

        enum_by_id = {
            value: key for key, value in self.enum_by_name.items()
        }
        
        str_results = []
        class_index = np.argmax(y_pred.T, axis=1)
        for i in range(len(x)):
            str_results.append(enum_by_id.get(class_index[i]))

        data[self.class_col] = str_results
        print(data)
        data.to_csv("houses.csv" , sep=",", index_label="Index", columns=[self.class_col])

    #### COMPUTATIONS
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
        # print(f"{a.shape} - {b.shape} - {self.biases.shape}")
        raw_result = self.weights.T @ x.T + self.biases
        # print(f"{raw_result=}")
        y_pred: np.ndarray = self.sigmoid(raw_result)
        # print(f"{self.y_pred=}")
        return y_pred

    def log_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Loss function or log loss, for visualization"""
        res: float = -(y * np.log(y_pred)
                       + (1 - y) * np.log(1 - y_pred)).mean()
        return res
    
    def compute_gradient(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray):
        """Uses derivative from log loss function, for gradient descent"""
        # res = ((self.y_pred - y) * x).mean()
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


    #### LOAD AND SAVE

    def save_weights(self):
        """Save weights to a file"""
        # Use json file
        # Save used feature for training and enum

        save_dict= {
            "class_enum": self.enum_by_name,
            "nb_classes": self.nb_classes,
            "nb_features": self.nb_features,
            "class_col": self.class_col,
            "trimeans": self.trimeans,
            "features_cols": self.features_cols,
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
        }

        model_folder: str = "models"
        model_file: str = datetime.now().strftime("DSLR_model_%Y-%m-%d_%H-%M-%S.json")
        model_path = os.path.join(model_folder, model_file)

        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        with open(model_path, 'w') as f:
            json.dump(save_dict, f, indent=4)

    def load_model(self):
        """Load a model from a file"""
        # Use json file
        pass

    ### TRACK
    # TODO: Add more figure for training stats visualisation (accuracy)

    def plot(self, data: list, name: str):
        plt.figure()
        plt.title(name)
        plt.plot(data)
        plt.show()

    ### DEBUG

    def print_all(self):
        print(f"{self.enum_by_name=}")
        print(f"{self.weights=}")
        print(f"{self.weights.shape=}")
        print(f"{self.biases=}")
        print(f"{self.biases.shape=}")
        print(f"{self.nb_classes=}, {self.nb_features=}")
