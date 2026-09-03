"""
HeatSentinel — Model B: PyTorch Bidirectional LSTM (Bi-LSTM) Sequence Model
Processes 14-day temporal weather windows to capture atmospheric inertia & multi-day heat momentum.
Outputs P_LSTM probability and 128-dim sequence embeddings.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PARQUET_PATH = DATA_DIR / "historical_weather_india.parquet"
MODEL_WEIGHTS_PATH = DATA_DIR / "bilstm_model.pt"

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class HeatwaveSequenceDataset(Dataset):
    """Sliding 14-day temporal window sequence dataset."""
    def __init__(self, df: pd.DataFrame, feature_cols: list, seq_len: int = 14):
        self.seq_len = seq_len
        self.X_seqs = []
        self.y_labels = []

        # Group chronologically per district
        for dist_code, group in df.groupby("district_code"):
            group = group.sort_values("date").reset_index(drop=True)
            data_matrix = group[feature_cols].values
            labels = group["is_heatwave"].values.astype(np.float32)

            for i in range(len(group) - seq_len):
                self.X_seqs.append(data_matrix[i : i + seq_len])
                self.y_labels.append(labels[i + seq_len - 1])

        self.X_seqs = torch.tensor(np.array(self.X_seqs), dtype=torch.float32)
        self.y_labels = torch.tensor(np.array(self.y_labels), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, idx):
        return self.X_seqs[idx], self.y_labels[idx]


class BiLSTMHeatwaveModel(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(BiLSTMHeatwaveModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.fc1(last_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        prob = self.sigmoid(out)
        return prob, last_step


class HeatSentinelBiLSTMPredictor:
    def __init__(self):
        self.features = [
            "temp_max", "humidity", "wind_speed", "solar_rad",
            "heat_index", "wbgt", "temp_anomaly",
            "rolling_mean_3d", "rolling_mean_7d", "rolling_max_14d",
            "consecutive_hot_days"
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMHeatwaveModel(input_dim=len(self.features), hidden_dim=128).to(self.device)
        self.is_trained = False
        self.train_on_historical_dataset(epochs=25)

    def train_on_historical_dataset(self, epochs: int = 25, batch_size: int = 64):
        if not PARQUET_PATH.exists():
            print(f"⚠️ Parquet dataset not found at {PARQUET_PATH}.")
            return

        print(f"🧠 Training PyTorch Bi-LSTM Model B for {epochs} Epochs on device [{self.device}]...")
        df = pd.read_parquet(PARQUET_PATH)

        dataset = HeatwaveSequenceDataset(df, self.features, seq_len=14)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                probs, _ = self.model(X_batch)
                loss = criterion(probs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.5f}")

        torch.save(self.model.state_dict(), MODEL_WEIGHTS_PATH)
        self.is_trained = True
        print(f"✅ Saved Bi-LSTM Model weights to {MODEL_WEIGHTS_PATH}")

    def predict_sequence(self, sequence_matrix: np.ndarray) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(sequence_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
            prob, embedding = self.model(x_tensor)
            return {
                "p_bilstm": round(float(prob.cpu().numpy()[0][0]), 4),
                "embedding": embedding.cpu().numpy()[0].tolist()
            }


bilstm_predictor = HeatSentinelBiLSTMPredictor()


if __name__ == "__main__":
    sample_seq = np.random.normal(35.0, 5.0, (14, 11))
    res = bilstm_predictor.predict_sequence(sample_seq)
    print("\n🧪 Bi-LSTM Test Prediction Output:")
    print(f"  P_LSTM: {res['p_bilstm']}")
    print(f"  Embedding Shape: {len(res['embedding'])} dimensions")

