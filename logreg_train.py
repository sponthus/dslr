import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from parsing import parse_logreg_train_args
from utils import standardise_data
import matplotlib.pyplot as plt

# To think about: Use of a config file for:
#   - dataset
#   - chosen features
#   - training parameters (cyles, learning rate)

class LogregTrain():

    def __init__(self):
        # Remove the saved dataframe?
        self.enum_by_name: dict| None = None
        self.nb_classes = 0
        self.nb_features = 0
        self.class_col: str | None = None
        self.features_cols: list[str] | None = None
        self.weights: np.ndarray | None = None
        self.biases: np.ndarray | None = None
        pass

    #### CONDITIONS

    def is_init(self) -> bool:
        print("coucou")
        if self.enum_by_name is None or self.nb_classes == 0 or self.nb_features == 0 or self.class_col is None or self.features_cols is None or self.weights is None or self.biases is None:
            print("re")
            return False
        return True

    def is_compatible(self, data: pd.DataFrame, training: bool):
        """Checks if a dataset is compatible with the class initialization
        and the class attributes validity."""
        columns = data.columns
        assert self.class_col in columns, f"'{self.class_col}' not in data"
        assert all(feature in columns for feature in self.features_cols), "not all features in data"
        assert self.nb_features == len(self.features_cols), "wrong nb_features"
        assert self.weights.shape == (self.nb_classes, self.nb_features), "wrong weights"
        assert self.biases.shape == (self.nb_classes, 1), "wrong biases"
        if training:
            assert self.nb_classes == len(data[self.class_col].unique()), "wrong nb_class"
            for data_class in data[self.class_col].unique():
                assert self.enum_by_name.get(data_class, False), "Unknown data_class"

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
        print(f"{y=}")

        losses = []
        for cycle in range(nb_cycles):
            y_pred: np.ndarray = self.predict(x)
            gradient_w, gradient_b = self.compute_gradient(x, y_pred, y)
            # print(f"{gradient_w=}, \n {gradient_b=}")
            logloss = self.log_loss(y, y_pred)
            losses.append(logloss)
            self.weights = self.update(self.weights, gradient_w, learning_rate)
            self.biases = self.update(self.biases, gradient_b, learning_rate)
            # print(f"{self.weights}")
        self.plot_loss(losses)

        y_validator = np.zeros((self.nb_classes, len(validator_data)))
        for i in range(len(validator_data)):
            class_index = int(validator_data.iloc[i][class_col])
            y_validator[class_index][i] = 1
        y_validator = np.argmax(y_validator, axis=0)
        validator_data = np.array(validator_data[self.features_cols])
    
        validation = np.argmax(self.predict(validator_data), axis=0)
        print(f"{validation=} / {y_validator=}\n")
        score = accuracy_score(y_true=y_validator, y_pred=validation)
        print(f"{score=}")

    def predictor(self, data: pd.DataFrame) -> np.ndarray:
        """Used to predict values from a trained model"""
        assert self.is_init(), "not initialized"
        assert self.is_compatible(data, training=False), "model training is not compatible with data"

        x: np.ndarray = np.array(data[self.features_cols])
        y_pred = self.predict(x)
        
        result = np.zeros((self.nb_classes, len(x)))
        for i in range(len(x)):
            class_index = np.argmax(y_pred.T[i])
            result[class_index][i] = 1

        # Create file with answers
        # Add function that translate results
        # Save results in a .csv file
        # print(f"{result=}")
        return result

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
        pass

    def load_model(self):
        """Load a model from a file"""
        # Use json file
        pass

    ### TRACK
    # TODO: Add more figure for training stats visualisation (accuracy)

    def plot_loss(self, losses: list):
        plt.figure()
        plt.plot(losses)
        plt.show()

    ### DEBUG

    def print_all(self):
        print(f"{self.enum_by_name=}")
        print(f"{self.weights=}")
        print(f"{self.biases=}")
        print(f"{self.nb_classes=}, {self.nb_features=}")
    
# TODO: Add main

def logreg_train(data: pd.DataFrame) -> None:
    chosen_cols = [
        "Muggle Studies",
        "History of Magic", "Transfiguration",
        "Divination",
        "Astronomy", "Herbology"
        ]
    class_col = "Hogwarts House"
    all_cols = chosen_cols + [class_col]
    data = data[all_cols]
    data = data.dropna(axis=0)
    print(data)
    test = LogregTrain()
    test.train(data, nb_cycles=1000, learning_rate=0.01, class_col=class_col, features_cols=chosen_cols)

def main():
    try:
        args: argparse.Namespace = parse_logreg_train_args()
    except Exception as e:
        print(f"Unexpected error: parse_describe_args(): {e}")
        sys.exit(1)

    try:
        logreg_train(args.dataset)
    except AssertionError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: describe(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
