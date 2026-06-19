import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from parsing import parse_logreg_train_args
from utils import standardise_data
import matplotlib.pyplot as plt


class LogregTrain():

    def __init__(
            self,
            training_data: pd.DataFrame,
            class_col: str,
            factor_col: list[str]
            ):
        """Initialize the logistic regression training class"""
        for col in factor_col:
            if col not in training_data.columns:
                raise AssertionError("")
        if class_col not in training_data.columns:
            raise AssertionError("")

        training_data = standardise_data(training_data)

        self.classes = training_data[class_col].unique()
        self.nb_classes = len(self.classes)
        self.nb_factors = len(factor_col)
        self.enum_by_name = {
            name: i for i, name in enumerate(self.classes)
        }
        self.enum_by_key = {
            i: name for i, name in enumerate(self.classes)
        }

        training_data[class_col] = \
            training_data[class_col].map(self.enum_by_name)
        self.test, self.validator = train_test_split(
            training_data,
            test_size=0.25,
            stratify=training_data[class_col],
            shuffle=True
        )
        print(f"{self.test=}\n{self.validator=}")

        # X = Factors values for each feature and sample
        self.x = np.array(self.test[factor_col])

        # Y = Expected class probability for each sample
        self.y = np.zeros((self.nb_classes, len(self.test)))
        for i in range(len(self.test)):
            class_index = int(self.test.iloc[i][class_col])
            self.y[class_index][i] = 1

        # TODO: Fix me to evaluate training
        # self.y_validator = np.zeros((self.nb_classes, len(self.validator)))
        # for i in range(len(self.validator)):
        #     class_index = int(self.validator.iloc[i][class_col])
        #     self.y[class_index][i] = 1
        # self.validator = np.array(self.validator[factor_col])

        self.y_pred = np.zeros((self.nb_classes, len(self.test)))

        # Weights = Initial weights for each factor and each class
        # Or np.random ?
        self.weights = np.full((self.nb_factors, self.nb_classes), 0.5)
        # One bias for each class because the biases are factorized in equation
        self.biases = np.zeros((self.nb_classes, 1))

        self.print_all()
        self.predict(self.x)

    def print_all(self):
        print(f"{self.enum_by_name=}")
        print(f"{self.weights=}")
        print(f"{self.biases=}")
        print(f"{self.x=}")
        print(f"{self.y=}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = self.weights.T
        b = x.T
        # print(f"{a.shape} - {b.shape} - {self.biases.shape}")
        raw_result = a @ b + self.biases
        # print(f"{raw_result=}")
        self.y_pred = self.sigmoid(raw_result)
        # print(f"{self.y_pred=}")
        result = np.zeros((self.nb_classes, len(self.test)))
        for i in range(len(x)):
            class_index = np.argmax(self.y_pred.T[i])
            result[class_index][i] = 1
        # print(f"{result=}")
        return result

    def plot_loss(self, losses: list):
        plt.figure()
        plt.plot(losses)
        plt.show()

    def train(self, nb_cycles: int, learning_rate: float):
        """Train the model with gradient descent"""
        losses = []
        for cycle in range(nb_cycles):
            # result = self.predict(self.x)
            self.predict(self.x)
            gradient_w, gradient_b = self.compute_gradient()
            # print(f"{gradient_w=}, \n {gradient_b=}")
            logloss = self.log_loss()
            losses.append(logloss)
            self.weights = self.update(self.weights, gradient_w, learning_rate)
            self.biases = self.update(self.biases, gradient_b, learning_rate)
            # print(f"{self.weights}")
        self.plot_loss(losses)
        # validation = self.predict(self.validator)
        # print(f"{validation=}\n{self.}")
        # score = accuracy_score(validation, self.y_validator)
        # print(f"{score=}")

    def save_weights(self):
        """Save weights to a file"""
        pass

    def load_model(self):
        """Load a model from a file"""
        pass

    def log_loss(self) -> float:
        """Loss function or log loss, for visualization"""
        res: float = -(self.y * np.log(self.y_pred)
                       + (1 - self.y) * np.log(1 - self.y_pred)).mean()
        return res

    def compute_gradient(self):
        """Uses derivative from log loss function, for gradient descent"""
        # res = ((self.y_pred - y) * x).mean()
        error = self.y_pred - self.y
        gradient_w = (self.x.T @ error.T) / self.x.shape[0]
        error_b = self.y_pred - self.y
        gradient_b = np.sum(error_b, axis=1, keepdims=True) / self.x.shape[0]
        return gradient_w, gradient_b

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function, turns any value to 0-1"""
        res: np.ndarray = 1 / (1 + np.exp(-x))
        return res

    def update(
            self,
            to_update: np.ndarray,
            gradient: np.ndarray,
            learning_rate: float
            ) -> np.ndarray:
        """Updates weights or bias with gradient modulated by learning_rate"""
        return to_update - (gradient * learning_rate)


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
    test = LogregTrain(data, class_col, chosen_cols)
    test.train(1000, 0.01)


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
