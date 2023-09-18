from pathlib import Path

import numpy as np
import pickle
import torch

TRAIN_CACHE = Path("train_nov22_no_trace.pkl")
TEST_CACHE = Path("test_nov22_no_trace.pkl")

NUM_EXAMPLES = 200
def build(props):
    DATA_FILE = TEST_CACHE if "test" in props else TRAIN_CACHE
    print(DATA_FILE)
    with open(DATA_FILE, 'rb') as file:
        dataset = pickle.load(file)

        X = dataset[:-1]
        Y = dataset[1:]

        X, Y = np.array(X), np.array(Y)
        X, Y = torch.tensor(X), torch.tensor(Y)
        X, Y = X.to(torch.float32), Y.to(torch.float32)
        rv = torch.utils.data.TensorDataset(X, Y)
        print(rv)
        return rv
    
if __name__ == "__main__":
    print("bom dia meninas")
    build([])
