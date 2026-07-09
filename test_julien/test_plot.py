import sys

import matplotlib.pyplot as plt
import pandas as pd
import json

def from_json_to_dataframe(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def add_bar_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], f"{y[i]:.4f}", ha='center')

def hist_data(data):
    labels = [", ".join(features) for features in data['features']]
    min_score = data['score'].min()
    max_score = data['score'].max()

    plt.figure(num="Feature Combination Scores", figsize=(20, 10))
    plt.bar(labels, data['score'])
    add_bar_labels(labels, data['score'].tolist())
    plt.title('Score by Feature Combination')
    plt.xlabel('Combination of Features')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylim(min_score - 0.01, max_score + 0.01)
    plt.tight_layout()
    plt.show()


def print_combination_scores(data):
    combination = [
        "Muggle Studies",
        "History of Magic", "Transfiguration",
        "Divination",
        "Astronomy", "Herbology"
        ]
    combination = sorted(combination)
    print(data[data['features'].apply(lambda x: sorted(x) == combination)])

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <results.json>")
        sys.exit(1)
    
    file_name = sys.argv[1]
    data = from_json_to_dataframe(file_name)
    data['features'] = data['features'].apply(sorted) 
    data = data.sort_values(by='score', ascending=False)
    
    print_combination_scores(data)
    data_filtered = data[data['score'] > 0.97]
    print(f"Best combinations: {data_filtered[:1]['features'].values[0]}")
    hist_data(data_filtered[:10])

if __name__ == "__main__":
    main()