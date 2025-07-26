import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.model.mtst import MTST  # update import path if needed
from utils.window import create_moving_window_dataset
from utils.data_processing import final_data_processing, load_data, preprocess_data
from utils.plots import plot_and_save_predictions, save_prediction_csv, plot_training_loss
import pandas as pd
import os
import matplotlib.pyplot as plt

# Model Hyperparameters
input_dim = 6
embed_dim = 64
heads = 8
dropout = 0.1
n_layers = 10
output_len = 5
input_len = 30
max_len = 5000

# Resolution Factors
high_res = 1
mid_res = 4
low_res = 10

# Training Hyperparameters
batch_size = 32
epochs = 100
lr = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_filename = "data/aave_train.csv"
test_filename = "data/aave_test.csv"

# Data Preprocessing
x_data, y_data = final_data_processing(train_filename, input_len, output_len, stride=1, fillna_method='ffill')
dataset = TensorDataset(x_data, y_data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model Initialization
model = MTST(
    input_dim=input_dim,
    embed_dim=embed_dim,
    heads=heads,
    dropout=dropout,
    n_layers=n_layers,
    output_len=output_len,
    max_len=max_len,
).to(device)

criterion = nn.MSELoss() # We can use MAE loss too as per https://arxiv.org/abs/2311.04147
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Training Loop
loss_tracker = []
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        preds = model(xb, high_res, mid_res, low_res)  # [B, output_len, 1]
        loss = criterion(preds, yb[:, :, 0])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_tracker.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

# Plot and save training loss
plot_training_loss(loss_tracker, filename="outputs/training_loss.png")

# Save the model
model_path = "outputs/mtst_model.pth"
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Inference on Test Data
df = pd.read_csv(test_filename, index_col=0)
df.fillna(method='ffill', inplace=True)
df = df[['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']]
df = df.apply(pd.to_numeric, errors='coerce')
data = df.values
input_data = data[-input_len:] 
x_input = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    pred = model(x_input, high_res, mid_res, low_res)
    pred = pred.squeeze(0).cpu().numpy()
    target = data[-(output_len):, 0] 

# Plot and save predictions
plot_and_save_predictions(target, pred, filename="outputs/prediction_plot.png")
save_prediction_csv(target, pred, filename="outputs/predictions.csv")
