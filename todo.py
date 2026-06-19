import numpy as np
import pandas as pd

class LogregTrain():

    def __init__(self):
        # Remove the saved dataframe?
        pass

    def train(
            self,
            data: pd.DataFrame,     # Add this
            nb_cycles: int,
            learning_rate: float
            ):

        # Check if model already train on a specific data
        # Chekc if model can train on the passed data
        # Store enum if not already present
        pass

    def save_weights(self):
        """Save weights to a file"""
        # Use json file
        # Save used feature for training and enum
        pass

    def load_model(self):
        """Load a model from a file"""
        # Use json file
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Refactored function
        pass

    def predictor(self, x: np.ndarray) -> np.ndarray:
        results = self.predict()
        # Add function that translate results
        # Save results in a .csv file

    # Move every stats calculation from class into different file?
    # Add more figure for training stats visualisation (accuracy)

# Use of a config file for:
#   - dataset
#   - chosen features
#   - training parameters (cyles, learning rate)
