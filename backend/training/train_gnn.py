import os
import sys
import json
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.stage1_gnn.st_gnn_model import SpatioTemporalGNN
from backend.stage1_gnn.icosahedral_mesh import SphericalIcosahedralMesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_gnn(epochs=5, lr=1e-3, seed=42, output_dir="runs/gnn"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = SphericalIcosahedralMesh(subdivision_level=2)
    edge_index = mesh.edge_index.to(device)
    edge_attr = mesh.edge_attr.to(device)
    num_nodes = mesh.num_nodes

    st_gnn = SpatioTemporalGNN(in_channels=6, hidden_channels=64, num_timesteps=9).to(device)
    optimizer = torch.optim.Adam(st_gnn.parameters(), lr=lr)
    criterion_pos = nn.MSELoss()
    criterion_int = nn.MSELoss()

    logger.info(f"Starting ST-GNN Training | Nodes: {num_nodes} | Edges: {edge_index.shape[1]} | Device: {device}")

    history = []
    st_gnn.train()
    for epoch in range(1, epochs + 1):
        # Create input sequence from mesh coordinates and physical fields
        x_seq = torch.zeros(9, num_nodes, 6, device=device)
        coords = torch.from_numpy(mesh.nodes_cartesian).float().to(device)
        x_seq[:, :, :3] = coords.unsqueeze(0).repeat(9, 1, 1)

        optimizer.zero_grad()
        pred_pos, pred_int = st_gnn(x_seq, edge_index, edge_attr)

        # Target trajectory displacements & intensities
        target_pos = torch.zeros(9, 2, device=device)
        target_int = torch.ones(9, 4, device=device) * 20.0

        loss_p = criterion_pos(pred_pos, target_pos)
        loss_i = criterion_int(pred_int, target_int)
        loss = loss_p + loss_i

        loss.backward()
        optimizer.step()

        logger.info(f"Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f} (Pos: {loss_p.item():.4f}, Int: {loss_i.item():.4f})")
        history.append({"epoch": epoch, "loss": loss.item()})

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    metrics_path = os.path.join(output_dir, "metrics.json")
    torch.save(st_gnn.state_dict(), ckpt_path)

    with open(metrics_path, "w") as f:
        json.dump({"seed": seed, "epochs": epochs, "final_loss": history[-1]["loss"], "history": history}, f, indent=2)

    logger.info(f"ST-GNN Training complete. Checkpoint saved to: {ckpt_path}")

def main():
    parser = argparse.ArgumentParser(description="StormTrace AI ST-GNN Model Trainer CLI")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/gnn")
    args = parser.parse_args()

    train_gnn(epochs=args.epochs, lr=args.lr, seed=args.seed, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
