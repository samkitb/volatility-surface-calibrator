import torch
import torch.nn as nn
import numpy as np

class VolSurfaceNet(nn.Module):
    """
    A feedforward neural network that learns the volatility surface.
    Input:  (moneyness, log_time_to_expiry) — 2 numbers
    Output: implied_volatility — 1 number
    basically learns the csv data to train the neural network allowing a user to input 
    a moneyness and time to expiry and get an implied volatility

    """
    def __init__(self, hidden_size=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.SiLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),

            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class VolSurfaceDataset(torch.utils.data.Dataset):
    """
    Packages your DataFrame into a format PyTorch can train on.
    """
    def __init__(self, df):
        self.X = torch.tensor(
            np.column_stack([
                df["moneyness"].values,
                np.log(df["time_to_expiry"].values)
            ]),
            dtype=torch.float32
        )
        self.y = torch.tensor(
            df["implied_vol"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def predict_surface(model, moneyness_grid, time_grid):
    """
    Given a trained model, predict IV across a dense grid.
    This produces the smooth surface for the 3D visualization.
    """
    model.eval()
    K, T = np.meshgrid(moneyness_grid, time_grid)
    K_flat = K.flatten()
    T_flat = T.flatten()

    X = torch.tensor(
        np.column_stack([K_flat, np.log(T_flat)]),
        dtype=torch.float32
    )

    with torch.no_grad():
        iv_flat = model(X).numpy()

    return iv_flat.reshape(K.shape)