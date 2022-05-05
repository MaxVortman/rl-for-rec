import pandas as pd
import pickle


def get_sequences(path):
    data = pd.read_csv(path)
    grouped_and_sorted = data.groupby("uid").apply(
        lambda x: list(x.sort_values(by=["date"])["sid"])
    )
    sequences = grouped_and_sorted.values
    return sequences


if __name__ == "__main__":
    path_prefix = "prepared_data/"
    data = {}
    for name in ["train", "validation_tr", "validation_te", "test_tr", "test_te"]:
        csv_path = path_prefix + name + ".csv"
        seq = get_sequences(csv_path)
        data[name] = seq
    with open(path_prefix + "seq_dataset.pkl", "wb") as f:
        pickle.dump(data, f)
