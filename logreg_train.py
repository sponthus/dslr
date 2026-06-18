import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from parsing import parse_logreg_train_args


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

        self.classes = training_data[class_col].unique()
        self.nb_classes = len(self.classes)
        self.nb_factors = len(factor_col)
        self.enum = {
            name: i for i, name in enumerate(self.classes)
        }

        self.test, self.validator = train_test_split(
            training_data, 
            test_size=0.25,
            stratify=training_data[class_col],
            shuffle=True
        )

        self.test[class_col] = self.test[class_col].map(self.enum)

        # X = Factors values for each feature and sample
        self.x = np.array(self.test[factor_col])

        # Y = Expected class probability for each sample
        self.y = np.zeros((self.nb_classes, len(self.test)))
        for i in range(len(self.test)):
            class_index = int(self.test.iloc[i][class_col])
            self.y[class_index][i] = 1

        # Weights = Initial weights for each factor and each class
        # Or np.random ?
        self.weights = np.full((self.nb_factors, self.nb_classes), 0.5)
        # One bias for each class because the biases are factorized in equation
        self.biases = np.zeros((self.nb_classes, 1))

        self.print_all()
        self.predict(self.x)

    def print_all(self):
        print(f"{self.enum=}")
        print(f"{self.weights=}")
        print(f"{self.biases=}")
        print(f"{self.x=}")
        print(f"{self.y=}")



    def predict(self, x: np.ndarray):
        a = self.weights.T
        b = x.T
        print(f"{a.shape} - {b.shape} - {self.biases.shape}")
        raw_result = a @ b + self.biases
        print(f"{raw_result=}")
        sig_result = self.sigmoid(raw_result)
        print(f"{sig_result=}")
        result = np.zeros((self.nb_classes, len(self.test)))
        for i in range(len(x)):
            class_index = self.enum.get(np.argmax(sig_result.T[i]))
            result[class_index][i] = 1
        print(f"{result=}")

    def train(self, nb_cycles: int, alpha: float):
        """Train the model with gradient descent"""
        for cycle in range(nb_cycles):
            pass
        pass

    def save_weights(self):
        """Save weights to a file"""
        pass

    def log_loss(self, y: np.ndarray, sigmoid: np.ndarray) -> float:
        """Loss function or log loss, for visualization"""
        res: float = -(y * np.log(sigmoid)
                       + (1 - y) * np.log(1 - sigmoid)).mean()
        return res

    def derivative(
            self,
            x: np.ndarray,
            y: np.ndarray,
            sigmoid: np.ndarray
            ) -> float:
        """Derivative from log loss function, for gradient descent"""
        res: float = ((sigmoid - y) * x).mean()
        return res

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function, turns any value to 0-1"""
        res: np.ndarray = 1 / (1 + np.exp(-x))
        return res

    def update(
            self,
            to_update: np.ndarray,
            gradient: float,
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
    test.train(2, 0.01)


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
