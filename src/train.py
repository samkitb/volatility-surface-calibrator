import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network import VolSurfaceNet, VolSurfaceDataset
from constraints import total_arbitrage_penalty

def train_vol_surface(ticker: str, epochs: int = 500, lr: float = 1e-3,
                      penalty_weight: float = 0.5):
    """
    Full training pipeline for a single ticker.
    Loads data, builds model, trains with arbitrage constraints, saves.
    """
    # --- Load data ---
    data_path = f"data/{ticker.lower()}_vol_surface.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Run fetch.py first to generate {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data points for {ticker}")

    # --- Dataset and DataLoader ---
    dataset = VolSurfaceDataset(df) #converts the csv data into a format that the neural network can train on

    loader = torch.utils.data.DataLoader( #loads the data in batches instead of all at once
        dataset, batch_size=32, shuffle=True
    )

    # --- Build model and optimizer ---
    model = VolSurfaceNet(hidden_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optimizer is the algorithm that updates the weights of the neural network

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(#reduces the learning rate if the training is not improving for 50 epochs
        optimizer, patience=50, factor=0.5
    )

    mse_loss = nn.MSELoss()

    # --- Training loop ---
    print(f"Training for {epochs} epochs...")
    history = []
    #heart of the training loop
    for epoch in range(epochs):
        model.train()
        epoch_data_loss = 0
        epoch_penalty = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad() #clears the gradients(how much the weights need to be updated) of the previous iteration

            # It takes the difference between each predicted IV and the real market IV, squares it, and averages across all 32. Squaring is important because it penalizes big errors much more than small ones. 
            y_pred = model(X_batch) 

            data_loss = mse_loss(y_pred, y_batch)# measure prediction error between the predicted and actual IV

            arb_penalty = total_arbitrage_penalty(model, weight=penalty_weight)# measure the penalty for the arbitrage free surface

            loss = data_loss + arb_penalty# combine the data loss and the arbitrage penalty

            loss.backward()#calculates the gradients of the loss with respect to the weights

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)#prevents the gradients from exploding (maintains speed and stability)

            optimizer.step()#updates the weights of the neural network

            epoch_data_loss += data_loss.item()
            epoch_penalty += arb_penalty.item()

        avg_data_loss = epoch_data_loss / len(loader)
        avg_penalty = epoch_penalty / len(loader)
        total = avg_data_loss + avg_penalty

        scheduler.step(total)
        history.append({
            "epoch": epoch,
            "data_loss": avg_data_loss,
            "arb_penalty": avg_penalty,
            "total": total
        })

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Data loss: {avg_data_loss:.6f} | "
                  f"Arb penalty: {avg_penalty:.6f} | Total: {total:.6f}")

    # --- Save model ---
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{ticker.lower()}_vol_surface.pt"
    torch.save({
        "model_state": model.state_dict(),
        "ticker": ticker,
        "hidden_size": 64,
        "epochs_trained": epochs,
        "final_loss": history[-1]["total"]
    }, save_path)

    print(f"\nModel saved to {save_path}")
    print(f"Final total loss: {history[-1]['total']:.6f}")

    return model, pd.DataFrame(history)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    model, history = train_vol_surface(ticker, epochs=500)