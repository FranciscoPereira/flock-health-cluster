"""
Simple PyTorch autoencoder for anomaly detection on 7-day ED visit windows.
"""
import torch
import torch.nn as nn
import io

class RespiratoryAutoencoder(nn.Module):
    """
    Autoencoder for 7-day sliding windows of respiratory ED visits.
    - Input: 7 days of visit counts
    - Learns normal patterns via reconstruction
    - Anomalies = high reconstruction error
    """
    def __init__(self, input_size=7, latent_dim=3):
        super(RespiratoryAutoencoder, self).__init__()

        # Encoder: 7 → 16 → 8 → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )

        # Decoder: latent_dim → 8 → 16 → 7
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)


def get_device():
    """Get appropriate device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def serialize_model(model: nn.Module) -> bytes:
    """Serialize model state dict to bytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_model(data: bytes, model: nn.Module = None) -> nn.Module:
    """Deserialize model from bytes."""
    if model is None:
        model = RespiratoryAutoencoder()
    state_dict = torch.load(io.BytesIO(data))
    model.load_state_dict(state_dict)
    return model
