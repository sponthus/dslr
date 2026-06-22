from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import standardise_data
import matplotlib.pyplot as plt

# To think about: Use of a config file for:
#   - dataset
#   - chosen features
#   - training parameters (cyles, learning rate)

class Logreg():

    def __init__(
            self,
            enum_by_name: dict | None = None,
            nb_classes = 0,
            nb_features = 0,
            class_col: str | None = None,
            features_cols: list[str] | None = None,
            weights: np.ndarray | None = None,
            biases: np.ndarray | None = None
            ):

        # Remove the saved dataframe?
        self.enum_by_name: dict | None = enum_by_name
        self.nb_classes = nb_classes
        self.nb_features = nb_features
        self.class_col: str | None = class_col
        self.features_cols: list[str] | None = features_cols
        self.weights: np.ndarray | None = weights
        self.biases: np.ndarray | None = biases

    @classmethod
    def from_file(cls, model_path: Path) -> Logreg:
        if not model_path.exists():
            raise FileNotFoundError(f"The file '{model_path}' does not exist.")

        if not model_path.is_file():
            raise FileNotFoundError(f"The path '{model_path}' is not a file.")
        
        with open(model_path, 'r') as f:
            json_str:str = f.read()
        
        data: dict = json.loads(json_str)
    
        # print(data)
        # print(type(data["class_enum"]))

        model = Logreg(
            data["class_enum"],
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
        if self.enum_by_name is None or self.nb_classes == 0 or self.nb_features == 0 or self.class_col is None or self.features_cols is None or self.weights is None or self.biases is None:
            return False
        return True

    def is_compatible(self, data: pd.DataFrame, training: bool):
        """Checks if a dataset is compatible with the class initialization
        and the class attributes validity."""
        columns = data.columns
        assert all(feature in columns for feature in self.features_cols), "not all features in data"
        assert self.nb_features == len(self.features_cols), "wrong nb_features"
        assert self.weights.shape == (self.nb_features, self.nb_classes), "wrong weights"
        assert self.biases.shape == (self.nb_classes, 1), "wrong biases"
        if training:
            assert self.class_col in columns, f"'{self.class_col}' not in data"
            assert self.nb_classes == len(data[self.class_col].unique()), "wrong nb_class"
            for data_class in data[self.class_col].unique():
                assert self.enum_by_name.get(data_class, False), "Unknown data_class"
        
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

    #### USAGE

    def train(
            self,
            data: pd.DataFrame,
            nb_cycles: int,
            learning_rate: float,
            class_col: str,
            features_cols: list[str]
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
        sd_data: pd.DataFrame = standardise_data(data)
        sd_data[class_col] = sd_data[class_col].map(self.enum_by_name)
        training_data, validator_data = train_test_split(
            sd_data,
            test_size=0.25,
            stratify=sd_data[class_col],
            shuffle=True
        )
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
        for cycle in range(nb_cycles):
            y_pred: np.ndarray = self.predict(x)
            gradient_w, gradient_b = self.compute_gradient(x, y_pred, y)
            # print(f"{gradient_w=}, \n {gradient_b=}")

            # For graphical representations
            logloss = self.log_loss(y, y_pred)
            losses.append(logloss)
            score = accuracy_score(y_true=np.argmax(y, axis=0), y_pred=np.argmax(y_pred, axis=0))
            scores.append(score)

            self.weights = self.update(self.weights, gradient_w, learning_rate)
            self.biases = self.update(self.biases, gradient_b, learning_rate)
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

    def predictor(self, data: pd.DataFrame) -> None:
        """Used to predict values from a trained model"""
        assert self.is_init(), "not initialized"
        assert self.is_compatible(data, training=False), "model training is not compatible with data"

        columns = ["Index"]
        columns.extend(self.features_cols)

        data.reset_index(inplace=True)
        data = data[columns].dropna(axis=0)
        data = data.set_index("Index")
        data.to_csv("initial.csv", sep=",", index_label="Index")
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
