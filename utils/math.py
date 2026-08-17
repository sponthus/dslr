import pandas as pd
import typing as tp
import numpy as np


class Percentiles(tp.NamedTuple):
    """A named tuple to hold the percentiles of a dataset"""
    min: float
    quartile_25: float
    quartile_50: float
    quartile_75: float
    max: float


def ft_percentiles(data: pd.Series, count: int) -> Percentiles:
    """Calculate the percentiles of a pandas Series"""
    if count == 0:
        raise ValueError("count can not be null")
    sorted_data: np.ndarray = np.sort(data)

    min: float = sorted_data[0]
    max: float = sorted_data[-1]

    if count % 2 == 0:
        quartile_25 = float(sorted_data[round(count * 1 / 4) - 1])
        quartile_50 = float(sorted_data[round(count * 1 / 2) - 1])
        quartile_75 = float(sorted_data[round(count * 3 / 4) - 1])
    else:
        quartile_25 = float(sorted_data[round((count + 1) * 1 / 4) - 1])
        quartile_50 = float(sorted_data[round((count + 1) * 1 / 2) - 1])
        quartile_75 = float(sorted_data[round((count + 1) * 3 / 4) - 1])

    percentiles = Percentiles(
        min=min,
        quartile_25=quartile_25,
        quartile_50=quartile_50,
        quartile_75=quartile_75,
        max=max
    )
    return percentiles


def ft_count(data: pd.Series) -> int:
    """Count the number of rows in a pandas Series"""
    count: int = 0
    for _ in data:
        count += 1
    return count


def ft_trimean(percentiles: Percentiles) -> float:
    """Trimean or Tukey's trimean, weighted average
    of distribution quartiles"""
    trimean: float = (
        percentiles.quartile_25
        + 2 * percentiles.quartile_50
        + percentiles.quartile_75
        ) / 4
    return trimean


def ft_mean(data: pd.Series, count: int) -> float:
    """Calculate the mean of a pandas Series"""
    if count == 0:
        raise ValueError("count can not be null")
    mean: float = float("NaN")
    total: float = float(0)

    for item in data:
        total += item

    mean = total / count

    return mean


def ft_deviation(data: pd.Series, mean: float, count: int) -> tuple:
    """Calculate the variance and standard deviation of a pandas Series"""
    if count == 0:
        raise ValueError("count can not be null")
    variance: float = (
        sum((x - mean) ** 2 for x in data)
        / count
    )
    std: float = variance ** 0.5
    return variance, std


def log_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Loss function or log loss, for visualization

    mean(-(y * log(y_pred)) + (1 - y) * log(1 - y_pred))
    """
    res: float = -(y * np.log(y_pred)
                   + (1 - y) * np.log(1 - y_pred)).mean()
    return res


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function, turns any value to 0-1"""
    res: np.ndarray = 1 / (1 + np.exp(-x))
    return res
