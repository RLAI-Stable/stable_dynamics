from pathlib import Path

import numpy as np
import torch
import pandas as pd


def build(props):
    print(props)
    DATA_FILE = Path(props["dataset"])
    N = int(props["n"])  # This should come from props again - for sine, 1 should work
    dataset = pd.read_pickle(DATA_FILE)

    # Extract X and Y
    X = np.array(dataset.iloc[:-N])  # excluding the last N rows (N-step)
    Y = np.array(dataset.iloc[N:])

    X, Y = torch.Tensor(X), torch.Tensor(Y)
    X, Y = X.to(torch.float64), Y.to(torch.float64)

    rv = torch.utils.data.TensorDataset(X, Y)
    return rv
