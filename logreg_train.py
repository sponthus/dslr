import numpy as np
import pandas as pd
from utils import get_data


class LogregTrain():

    def __init__(
            self,
            training_data: pd.DataFrame,
            class_col: str,
            factor_col: list[str]
            ):
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

        # TODO: Divide data here

        print(f"{self.enum=}")
        training_data[class_col] = training_data[class_col].map(self.enum)
        self.training_data = training_data

        # X = Factors values for each feature and sample
        self.x = np.array(training_data[factor_col])

        # Y = Expected class probability for each sample
        self.y = np.zeros((self.nb_classes, len(training_data)))
        for i in range(len(training_data)):
            class_index = int(training_data.iloc[i][class_col])
            print(class_index)
            self.y[class_index][i] = 1

        # Weights = Initial weights for each factor and each class
        # Or np.random ?
        self.weights = np.full((self.nb_factors, self.nb_classes), 0)
        # print(self.weights)

    def train(self, nb_cycles: int, alpha: float):
        for cycle in range(nb_cycles):

            pass
        pass

    def save_weights(self):
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


data: pd.DataFrame = get_data("datasets/dataset_train.csv")
chosen_cols = ["Muggle Studies",
               "History of Magic", "Transfiguration",
               "Divination",
               "Astronomy", "Herbology",
               ]
class_col = "Hogwarts House"
all_cols = chosen_cols + [class_col]
data = data[all_cols]
data = data.dropna(axis=0)
print(data)
test = LogregTrain(data, class_col, chosen_cols)
