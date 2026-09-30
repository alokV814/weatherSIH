"""
StormTrace AI - Training Script for PyTorch Spherical ST-GNN Model 
Executes forward pass, loss computation, backward pass, and optimizer.step()
"""
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.stage1_gnn.model import PyTorchSTGNNModel
from backend.stage1_gnn.dataset import WeatherSphericalGraphDataset
from backend.stage1_gnn.icosahedral_mesh import build_spherical_icosahedral_mesh


def train_st_gnn(epochs: int = 5, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ST-GNN Training] Initializing PyTorch training loop on device: {device}")

    # Build spherical mesh edge index
    mesh = build_spherical_icosahedral_mesh(level=1)
    edge_index = mesh["edge_index"].to(device)

    dataset = WeatherSphericalGraphDataset(num_samples=10, timesteps=9, num_nodes=mesh["num_nodes"])
    model = PyTorchSTGNNModel(in_channels=6, hidden_dim=64, out_channels=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_seq, y_track, y_intensity in dataset:
            x_seq = x_seq.to(device)
            y_track = y_track.to(device)
            y_intensity = y_intensity.to(device)

            optimizer.zero_grad()
            pred_track, pred_intensity = model(x_seq, edge_index)

            loss_track = F.mse_loss(pred_track, y_track)
            loss_intensity = F.mse_loss(pred_intensity, y_intensity)
            loss = loss_track + 0.1 * loss_intensity

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        history.append({"epoch": epoch, "loss": round(avg_loss, 4)})
        print(f"[Epoch {epoch:02d}/{epochs}] ST-GNN Training Loss: {avg_loss:.4f}")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, "st_gnn_checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[ST-GNN Training] Checkpoint saved successfully to {ckpt_path}")
    return {"status": "trained", "checkpoint_path": ckpt_path, "final_loss": history[-1]["loss"]}

if __name__ == "__main__":
    train_st_gnn(epochs=3)
