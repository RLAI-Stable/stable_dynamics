from pathlib import Path

import numpy as np
import pickle
import torch

TRAIN_FILE = Path("train_nov22_no_trace.pkl")
TEST_FILE = Path("test_nov22_no_trace.pkl")

SENSOR_INDEX = 53
N = 10

def build(props):
    DATA_FILE = TEST_FILE if "test" in props else TRAIN_FILE
    N = int(props["n"]) if "n" in props else 10
    with open(DATA_FILE, 'rb') as file:
        dataset = pickle.load(file)

        # Extract X and Y
        X = np.array(dataset.iloc[:-N])  # excluding the last N rows (N-step)
        Y = np.array(dataset.iloc[N:])

        X, Y = torch.tensor(X), torch.tensor(Y)
        X, Y = X.to(torch.float32), Y.to(torch.float32)

        rv = torch.utils.data.TensorDataset(X, Y)
        return rv
    
if __name__ == "__main__":
    build([])