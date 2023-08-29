from datasets.mountaincar import get_env, build
from models import stabledynamics as sd
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from util import to_variable

def lossfn(Ypred, Yactual, X):
    return torch.nn.functional.mse_loss(Ypred, Yactual)

props = {
    "mountain_car": True,

    'force': 0.001,
    'gravity': 0.0025,
    'num_examples': 200,

    'latent_space_dim': 3,
    'projfn': 'PSD-REHU',

    'num_epochs': 1000,
    'batch_size': 100,
    'learning_rate': 1e-3,

}

# Build the dataset:
data = build(props)
dataset_loader = DataLoader(data, batch_size=props['batch_size'], shuffle=True)

# Build the model:
sd.configure(props)
model = sd.model
print(model)

# Train the model:

if torch.cuda.is_available():
    model.cuda()
opt = optim.Adam(model.parameters(), lr=props['learning_rate'])

for epoch in range(1, props['num_epochs'] + 1):
    model.train()
    loss_parts = []

    for batch_idx, data in enumerate(dataset_loader):
            opt.zero_grad()
            X, Yactual = data
            
            X = to_variable(X, cuda=torch.cuda.is_available())
            Yactual = to_variable(Yactual, cuda=torch.cuda.is_available())

            Ypred = model(X)
            print(Ypred.shape, Yactual.shape, X.shape)
            loss = lossfn(Ypred, Yactual, X)
            
            loss_parts.append(np.array([l.cpu().item() for l in loss]))

            optim_loss = loss[0] if isinstance(loss, (tuple, list)) else loss
            optim_loss.backward()
            opt.step()
    print(f"Epoch {epoch}: {sum(loss_parts) / len(loss_parts)}")

# Save the model:
torch.save(model.state_dict(), "model.pt")
print(model.state_dict())


