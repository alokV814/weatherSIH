"""
StormTrace AI - PyTorch ST-GNN Smoke Training Test
Verifies that the PyTorch Spherical ST-GNN executes:
1. Graph construction
2. Forward pass
3. Loss calculation
4. Backward pass (gradients computed)
5. Optimizer step
6. Verification that parameters and loss change
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.stage1_gnn.model import PyTorchSTGNNModel

from backend.stage1_gnn.icosahedral_mesh import build_spherical_icosahedral_mesh

def test_st_gnn_training_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = build_spherical_icosahedral_mesh(level=1)
    edge_index = mesh["edge_index"].to(device=device, dtype=torch.long)


    model = PyTorchSTGNNModel(in_channels=6, hidden_dim=64, out_channels=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Initial parameter norm snapshot
    init_param_norm = sum(p.norm().item() for p in model.parameters())

    x_seq = torch.randn(9, mesh["num_nodes"], 6, device=device)
    y_track = torch.randn(9, 2, device=device)
    y_intensity = torch.randn(9, 1, device=device)

    # Step 1: Forward pass
    pred_track, pred_intensity = model(x_seq, edge_index)
    loss1 = F.mse_loss(pred_track, y_track) + F.mse_loss(pred_intensity, y_intensity)

    # Step 2: Backward pass + Optimizer Step
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    # Step 3: Second forward pass to verify optimization
    pred_track2, pred_intensity2 = model(x_seq, edge_index)
    loss2 = F.mse_loss(pred_track2, y_track) + F.mse_loss(pred_intensity2, y_intensity)

    updated_param_norm = sum(p.norm().item() for p in model.parameters())

    print(f"[Smoke Test] Initial Loss: {loss1.item():.4f} -> Post-Step Loss: {loss2.item():.4f}")
    assert loss2.item() < loss1.item(), "Loss should decrease after optimizer step"
    assert updated_param_norm != init_param_norm, "Model parameters must update after optimizer step"
    print("[PASS] PyTorch ST-GNN Smoke Training Test Passed Successfully!")

if __name__ == "__main__":
    test_st_gnn_training_smoke()
