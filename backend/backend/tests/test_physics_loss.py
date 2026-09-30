import os
import sys
import torch
import torch.nn.functional as F

# Ensure backend directory is in sys.path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from stage2_diffusion.physics_loss import compute_physics_loss_with_breakdown

def test_physics_conservation_loss():
    print("Testing DDPM Physics Loss (Mass and Moisture Conservation)...")
    
    # 1. Setup mock tensors simulating 12km coarse input and 5km DDPM prediction
    B, C, H, W = 2, 1, 64, 64
    pred_high_res = torch.ones(B, C, H, W) * 100.0  # e.g. 100 mm/h peak
    coarse_input = torch.ones(B, C, H//2, W//2) * 20.0 # 20 mm/h average
    
    # Mock physical fields
    u = torch.randn(B, C, H, W) * 5.0
    v = torch.randn(B, C, H, W) * 5.0
    q = torch.rand(B, C, H, W) * 0.02
    T = torch.randn(B, C, H, W) * 2.0 + 300.0
    
    # 2. Compute loss
    res = compute_physics_loss_with_breakdown(pred_high_res, coarse_input, u, v, q, T)
    breakdown = res["breakdown"]
    
    # 3. Assertions
    mass_loss = breakdown["massConservationLoss"]
    moisture_loss = breakdown["moistureFluxLoss"]
    
    print(f"Mass Conservation Loss: {mass_loss}")
    print(f"Moisture Flux Loss: {moisture_loss}")
    
    # Mass loss should be > 0 because pred mean (100) != coarse mean (20)
    assert mass_loss > 1000.0, f"Mass conservation not enforced correctly, loss={mass_loss}"
    
    # Moisture loss should be > 0 because 100mm/h > div_flux * 10
    assert moisture_loss > 50.0, f"Moisture conservation not enforced correctly, loss={moisture_loss}"
    
    print("All physics loss conservation tests passed!")

if __name__ == "__main__":
    test_physics_conservation_loss()
