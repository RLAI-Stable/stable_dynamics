from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from sympy import Dummy, lambdify, symbols
from sympy.physics import mechanics

from gym.envs.classic_control import MountainCarEnv


CACHE = Path("pendulum-cache/")

def get_env(force=0.001, gravity=0.0025):
    
    def MCEnvStep(state, action):
        '''
        Equations taken from OpenAI Gym's MountainCarEnv class.
        https://www.gymlibrary.dev/environments/classic_control/mountain_car/
        '''
        assert state.shape == (2,), f"Expected state shape (2,), got {state.shape}"
        assert action.shape == (1,), f"Expected action shape (1,), got {action.shape}"
        assert action in [0, 1, 2], f"Expected action in [0, 1, 2], got {action}"
        # 0 - left, 1 - nothing, 2 - right
        vel, pos = state
        vel += (action-1)*force - np.cos(3*pos)*gravity
        pos += vel
        
        return np.array([vel, pos])
    
    return MCEnvStep

def _redim(inp):
    vec = np.array(inp)
    # Wrap all dimensions:
    n = vec.shape[1] // 2
    assert vec.shape[1] == n*2

    # Get angular positions:
    pos = vec[:,:n]
    l = 100

    if np.any(pos < -np.pi):
        # In multiples of 2pi
        adj, _ = np.modf((pos[pos < -np.pi] + np.pi) / (2*np.pi))
        # Scale it back
        pos[pos < -np.pi] = (adj * 2*np.pi) + np.pi
        assert not np.any(pos < -np.pi)

    if np.any(pos >= np.pi):
        # In multiples of 2pi
        adj, _ = np.modf((pos[pos >= np.pi] - np.pi) / (2*np.pi))
        # Scale it back
        pos[pos >= np.pi] = (adj * 2*np.pi) - np.pi
        assert not np.any(pos >= np.pi)

    vec[:,:n] = pos
    return vec

NUM_EXAMPLES = 200
def build(props: dict) -> torch.utils.data.TensorDataset:
    global NUM_EXAMPLES
    NUM_EXAMPLES = props.get("num_examples", 200)
    force = props.get("force", 0.001)
    gravity = props.get("gravity", 0.0025)

    MCenv = get_env(force, gravity)
    cache_path = CACHE / f"mc-{force}-{gravity}.npz"

    if not cache_path.exists(): 
        states = np.random.rand(NUM_EXAMPLES,2).astype(np.float32)
        actions = np.random.randint(0, 3, size=(NUM_EXAMPLES,1)).astype(np.float32)
        X = np.concatenate([states, actions], axis=1)

        # Calculate next state using MCenv
        Y = np.array([MCenv(x, a) for x, a in zip(states, actions)]).reshape(-1, 2)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, Y=Y)
    else:
        load = np.load(cache_path)
        X = load["X"]
        Y = load["Y"]

    rv = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    rv._env = MCenv
    rv._force = force
    rv._gravity = gravity
    rv._num_examples = NUM_EXAMPLES
    rv._redim = _redim
    return rv

if __name__ == "__main__":
    pass