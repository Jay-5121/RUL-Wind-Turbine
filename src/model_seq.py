import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
from typing import Dict, Tuple, Any, List
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sequence_model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_stack = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        return self.fc_stack(out[:, -1, :]).squeeze()

class WindTurbineDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.targets[idx]

def make_windows(df: pd.DataFrame, seq_len: int, t_id_col: str, f_cols: List[str], t_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seqs, targs, t_ids = [], [], []
    for name, group in df.groupby(t_id_col):
        if len(group) < seq_len: continue
        feats, targs_data = group[f_cols].values, group[t_col].values
        for i in range(len(group) - seq_len + 1):
            seqs.append(feats[i:i + seq_len])
            targs.append(targs_data[i + seq_len - 1])
            t_ids.append(name)
    return np.array(seqs), np.array(targs), np.array(t_ids)

def train_gru_model(
    features_path: str = "data/features.parquet",
    turbine_id_col: str = "turbine_id",
    rul_col: str = "rul_hours",
    sequence_length: int = 10, # FIX: Reduced sequence length for small dataset
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 64,
    model_path: str = "models/rul_gru.pth",
    scaler_path: str = "models/gru_scaler.pkl"
):
    logger.info("Starting GRU sequence model training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_parquet(features_path).dropna(subset=[rul_col])
    
    feature_names = [col for col in df.select_dtypes(include=np.number).columns if col not in [turbine_id_col, rul_col]]
    sequences, targets, _ = make_windows(df, sequence_length, turbine_id_col, feature_names, rul_col)

    if sequences.shape[0] == 0:
        logger.error(f"No sequences were created with sequence_length={sequence_length}. The dataset is too small after feature engineering.")
        raise ValueError("Cannot train GRU model on an empty dataset. Increase data size or decrease sequence_length.")

    sequences[np.isinf(sequences)] = np.finfo(np.float32).max

    scaler = StandardScaler()
    sequences_norm = scaler.fit_transform(sequences.reshape(-1, len(feature_names))).reshape(sequences.shape)

    loader = DataLoader(WindTurbineDataset(sequences_norm, targets), batch_size=batch_size, shuffle=True)
    model = GRUModel(len(feature_names), hidden_size, num_layers, dropout).to(device)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info(f"Training GRU model on {sequences.shape[0]} sequences...")
    for epoch in range(epochs):
        model.train()
        for seq, tar in loader:
            seq, tar = seq.to(device), tar.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq), tar)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0: logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'feature_names': feature_names}, model_path)
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
        
    logger.info(f"Model and scaler saved.")

if __name__ == "__main__":
    train_gru_model()
