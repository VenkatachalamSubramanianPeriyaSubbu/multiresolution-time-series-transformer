import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_training_loss(loss_tracker, filename="outputs/training_loss.png", show=False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_tracker, label="Training Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved training loss plot to {filename}")

def plot_and_save_predictions(y_true, y_pred, filename="outputs/prediction_plot.png"):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Ground Truth", linewidth=2)
    plt.plot(y_pred, label="Prediction", linestyle='--')
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("MTST Prediction vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved prediction plot to {filename}")
    plt.close()

def save_prediction_csv(y_true, y_pred, filename="outputs/predictions.csv"):
    df = pd.DataFrame({
        "Ground Truth": y_true,
        "Prediction": y_pred
    })
    df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")
