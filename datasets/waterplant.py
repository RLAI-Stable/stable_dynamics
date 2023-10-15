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
        Y = np.zeros_like(X)             # Y is .zeros() since we only care about one sensor in prediction

        # Populate Y with the specified conditions
        for i in range(len(Y)):
            # TODO: Change this to mean X_t + 1 not X_t
            Y[i, SENSOR_INDEX] = np.sum(dataset.iloc[i+1:i+N+1, SENSOR_INDEX])

        X, Y = torch.tensor(X), torch.tensor(Y)
        X, Y = X.to(torch.float32), Y.to(torch.float32)

        rv = torch.utils.data.TensorDataset(X, Y)
        return rv
    
if __name__ == "__main__":
    build([])