import pandas as pd
import os

datapath = os.path.join(os.path.dirname(__file__), 'data')

def get_datasets():
    return [os.path.splitext(filename)[0] for filename in os.listdir(datapath)]


def load_dataset(datasetname, normalize):
    filepathname = os.path.join(datapath, datasetname + ".csv")
    df = pd.read_csv(filepathname)
    X = df[[c for c in df.columns if c != "label"]].values
    if normalize:
        X = (X - X.mean()) / X.std()

    y = df["label"].values
    y[y == "nominal"] = -1
    y[y == "anomaly"] = 1

    return X, y
