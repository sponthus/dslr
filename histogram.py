import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def get_data(path: Path) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(path, sep=",")
    if "Index" in data.columns:
        data = data.set_index("Index")
    return data


data = get_data(path="datasets/dataset_train.csv")
numeric_col = [col for col in data.columns if data[col].dtype == float]

COLORS_HOUSES = {
    "Slytherin": (0, 1, 0), 
    "Gryffindor": (1, 0, 0), 
    "Ravenclaw": (0, 0, 1), 
    "Hufflepuff": (1, 1, 0)
}

plt.figure()
for col in numeric_col :
    for house, color in COLORS_HOUSES.items():
        color_w_transparency = (*color, 0.5)
        plt.hist(
            x=data[data["Hogwarts House"] == house][col],
            fc=color_w_transparency,
            label=house
        )
        plt.xlabel("Mark")
        plt.ylabel("Number of students")
        plt.legend()
    plt.title(col)
    plt.show()