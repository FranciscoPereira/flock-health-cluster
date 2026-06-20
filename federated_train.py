"""
Federated learning: 3 NHS nodes train locally, aggregate globally via Flock.io.
DATA STAYS LOCAL - only model weights are shared.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from autoencoder import RespiratoryAutoencoder, serialize_model, deserialize_model, get_device


class NodeLocalTrainer:
    """Represents one NHS node's local training."""

    def __init__(self, node_name: str, csv_path: str, device=None):
        self.node_name = node_name
        self.csv_path = csv_path
        self.device = device or get_device()
        self._load_data()

    def _load_data(self):
        """Load local CSV data (STAYS ON THIS NODE)."""
        df = pd.read_csv(self.csv_path)
        visits = df["respiratory_visits"].values.astype(np.float32)

        # Use only pre-outbreak data (days 1-44) for training
        # This ensures model learns normal patterns, not the outbreak
        train_visits = visits[:44]

        # Create 7-day sliding windows
        windows = []
        for i in range(len(train_visits) - 6):
            window = train_visits[i:i+7]
            # Normalize to 0-1 range (use full dataset range for consistency)
            window_norm = (window - visits.min()) / (visits.max() - visits.min() + 1e-6)
            windows.append(window_norm)

        self.windows = np.array(windows)
        self.data_tensor = torch.tensor(self.windows, dtype=torch.float32).to(self.device)
        print(f"  {self.node_name}: Loaded {len(self.windows)} windows (pre-outbreak, days 1-44)")
        print(f"    └─ Data stays LOCAL (never sent to other nodes)")

    def train_locally(self, model: RespiratoryAutoencoder, epochs: int = 3, lr: float = 0.01) -> RespiratoryAutoencoder:
        """
        Train model on LOCAL data.
        Only model weights returned - NO RAW DATA SHARED.
        """
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            losses = []
            for window in self.data_tensor:
                optimizer.zero_grad()
                reconstructed = model(window.unsqueeze(0))
                loss = criterion(reconstructed, window.unsqueeze(0))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            if epoch == epochs - 1:
                print(f"    └─ Training complete | Final loss: {avg_loss:.4f}")

        return model

    def get_model_weights(self, model: RespiratoryAutoencoder) -> bytes:
        """Export model weights as bytes (safe to share)."""
        return serialize_model(model)


class FederatedAggregator:
    """Aggregates model updates from multiple nodes (Flock.io pattern)."""

    @staticmethod
    def aggregate_weights(weights_list: list[bytes]) -> bytes:
        """
        FedAvg: Average model weights across nodes.
        Each node only contributed trained weights, no data leakage.
        """
        models = [deserialize_model(w) for w in weights_list]

        # Simple averaging: model_avg = (model_0 + model_1 + model_2) / 3
        avg_state = models[0].state_dict()
        for key in avg_state.keys():
            avg_state[key] = avg_state[key].clone()
            for i in range(1, len(models)):
                avg_state[key] += models[i].state_dict()[key]
            avg_state[key] /= len(models)

        # Save aggregated model
        global_model = RespiratoryAutoencoder()
        global_model.load_state_dict(avg_state)
        return serialize_model(global_model)


def federated_training_loop(rounds: int = 2) -> bytes:
    """
    Simulate federated learning across 3 NHS nodes.

    Flow:
    1. Each node trains locally on its own data
    2. Nodes send only model weights (NOT data) to aggregator
    3. Aggregator averages weights
    4. Repeat for multiple rounds
    """
    device = get_device()
    print(f"\n{'='*70}")
    print("FEDERATED LEARNING: 3 NHS Nodes (Data stays LOCAL)")
    print(f"{'='*70}")

    node_configs = [
        ("node_a", "data/node_a.csv"),
        ("node_b", "data/node_b.csv"),
        ("node_c", "data/node_c.csv"),
    ]

    # Initialize trainers
    trainers = [NodeLocalTrainer(name, csv, device) for name, csv in node_configs]

    # Federated training rounds
    global_model = RespiratoryAutoencoder().to(device)
    global_weights = serialize_model(global_model)

    for round_num in range(rounds):
        print(f"\n--- FEDERATED ROUND {round_num + 1}/{rounds} ---")

        # Step 1: Each node trains locally (data stays on node)
        local_weights = []
        for trainer in trainers:
            print(f"→ {trainer.node_name}: Training locally...")
            model = deserialize_model(global_weights, RespiratoryAutoencoder().to(device))
            trained_model = trainer.train_locally(model, epochs=3)
            weights = trainer.get_model_weights(trained_model)
            local_weights.append(weights)
            print(f"  {trainer.node_name}: Weights exported (no data leaked)")

        # Step 2: Aggregate weights (via Flock.io aggregation)
        print(f"\n→ AGGREGATING weights from 3 nodes...")
        global_weights = FederatedAggregator.aggregate_weights(local_weights)
        print(f"  ✓ Global model updated via FedAvg")
        print(f"  ✓ KEY PROPERTY: Raw patient data never left any node")

    print(f"\n{'='*70}")
    print("✓ FEDERATED TRAINING COMPLETE")
    print(f"{'='*70}\n")
    return global_weights


if __name__ == "__main__":
    global_weights = federated_training_loop(rounds=2)

    # Save trained global model
    with open("models/global_model.pt", "wb") as f:
        f.write(global_weights)
    print(f"✓ Global model saved to models/global_model.pt")
