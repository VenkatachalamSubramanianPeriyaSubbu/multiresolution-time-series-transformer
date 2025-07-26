import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.model.mtst import MTST  # update import path if needed
from utils.window import create_moving_window_dataset
from utils.synthetic_data import generate_synthetic_data 

# Model Hyperparameters
input_dim = 1
embed_dim = 64
heads = 8
dropout = 0.1
n_layers = 4
output_len = 24
input_len = 120
max_len = 5000

# Resolution Factors
high_res = 1
mid_res = 4
low_res = 10

# Training Hyperparameters
batch_size = 32
epochs = 50
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic Data Generation
series = generate_synthetic_data()

# Create windowed dataset
x_data, y_data = create_moving_window_dataset(series, input_len, output_len, stride=1)
print(f"Training samples: {x_data.shape[0]}, Input shape: {x_data.shape[1:]}, Target shape: {y_data.shape[1:]}")

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
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        preds = model(xb, high_res, mid_res, low_res)  # [B, output_len, 1]
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
