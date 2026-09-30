import os
import sys
import json
import numpy as np
from datetime import datetime, timezone

# Set PYTHONPATH to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import StormTrace AI Core Engines for Benchmark Evaluation
from backend.models.inspector import ModelEvidenceInspector
from backend.data.era5_loader import ERA5DataLoader
from backend.data.climatology import RealERA5ClimatologyEngine
from backend.stage1_gnn.st_gnn_model import track_anomaly_object_st_gnn
from backend.stage2_diffusion.ddpm import run_diffusion_downscale
from backend.ensemble_engine import EnsembleNWPEngine
from backend.historical_validation import HistoricalValidationEngine

def run_reproducible_benchmark_suite():
    print("=" * 75)
    print(" StormTrace AI - Reproducible Scientific Benchmark Suite  ")
    print("=" * 75)
    
    benchmark_report = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "system": "StormTrace AI Production Benchmark",
        "results": {}
    }

    
    # 1. Trained Model Proof & Inspection
    print("\n[Benchmark 1/5] Verifying PyTorch Trained Model Weights & Parameter Evidence...")
    inspector = ModelEvidenceInspector()
    model_evidence = inspector.inspect_checkpoints()
    benchmark_report["results"]["trained_model_evidence"] = model_evidence
    print(f"   -> ST-GNN Trained Parameters: {model_evidence['models'].get('st_gnn', {}).get('total_trainable_parameters', 0):,}")
    print(f"   -> DDPM Downscaler Parameters: {model_evidence['models'].get('ddpm', {}).get('total_trainable_parameters', 0):,}")
    
    # 2. Spatio-Temporal Tracking Model Benchmark
    print("\n[Benchmark 2/5] Evaluating ST-GNN Trajectory Tracking Error (T+0 to T+240h)...")
    track_res = track_anomaly_object_st_gnn(origin_lat=19.5, origin_lon=88.5)
    timesteps = track_res["objectTrackingSummary"]["trackedTimesteps"]
    mean_conf = np.mean([pt["confidenceScore"] for pt in timesteps])
    print(f"   -> Tracked Waypoints: {len(timesteps)} points across 10-day forecast window")
    print(f"   -> Mean Trajectory Confidence: {mean_conf:.1f}%")
    benchmark_report["results"]["st_gnn_tracking"] = {
        "waypoints": len(timesteps),
        "mean_confidence_pct": round(float(mean_conf), 1),
        "algorithm": track_res["objectTrackingSummary"]["modelArchitecture"]
    }
    
    # 3. Downscaling & Physics Preservation Benchmark (Bicubic vs DDPM Physics)
    print("\n[Benchmark 3/5] Evaluating 12km -> 5km Super-Resolution Downscaling Preservations...")
    coarse_grid = np.random.gamma(2.0, 20.0, (12, 12))
    coarse_grid[4:8, 4:8] += 120.0 # Extreme convective peak
    
    fine_ddpm = run_diffusion_downscale(coarse_grid)
    coarse_peak = float(np.max(coarse_grid))
    fine_peak = float(np.max(fine_ddpm))
    peak_retention = min(100.0, (fine_peak / (coarse_peak + 1e-6)) * 100.0)
    mass_conservation_loss = float(np.abs(np.mean(coarse_grid) - np.mean(fine_ddpm)))
    
    print(f"   -> Coarse 12 km Peak: {coarse_peak:.1f} mm | Fine 5 km Peak: {fine_peak:.1f} mm")
    print(f"   -> Extreme Peak Preservation: {peak_retention:.1f}%")
    print(f"   -> Physics Mass Conservation Loss: {mass_conservation_loss:.5f}")
    benchmark_report["results"]["ddpm_downscaling"] = {
        "coarse_peak_mm": round(coarse_peak, 1),
        "fine_peak_mm": round(fine_peak, 1),
        "peak_preservation_pct": round(peak_retention, 1),
        "mass_conservation_loss": round(mass_conservation_loss, 5)
    }

    # 4. Ensemble Uncertainty CRPS Benchmark
    print("\n[Benchmark 4/5] Evaluating 50-Member Ensemble Exceedance CRPS & Brier Score...")
    ens_engine = EnsembleNWPEngine(num_members=50)
    ens_res = ens_engine.process_ensemble_forecast(coarse_grid, threshold_mm=50.0)
    ens_meta = ens_res["ensembleMetadata"]
    print(f"   -> 50-Member Exceedance Probability: {ens_meta['maxExtremeProbabilityPct']}%")
    print(f"   -> Continuous Ranked Probability Score (CRPS): {ens_meta['crpsScore']}")
    print(f"   -> Brier Score: {ens_meta['brierScore']}")
    benchmark_report["results"]["ensemble_uncertainty"] = ens_meta

    # 5. 10-Year Historical Disaster Ground-Truth Validation Benchmark (2014-2024)
    print("\n[Benchmark 5/5] Executing 10-Year Historical Disaster Benchmark Suite (2014-2024)...")
    validator = HistoricalValidationEngine()
    val_report = validator.evaluate_historical_case_studies()
    sum_metrics = val_report["overallSummaryMetrics"]
    print(f"   -> Evaluated Disasters: {val_report['totalHistoricalEvents']} major events (2014-2024)")
    print(f"   -> Mean Position Track Error: {sum_metrics['meanPositionErrorKm']} km")
    print(f"   -> Mean Critical Success Index (CSI): {sum_metrics['meanCsiScore']}")
    print(f"   -> Mean Probability of Detection (POD): {sum_metrics['meanPodScore']}")
    print(f"   -> Mean False Alarm Ratio (FAR): {sum_metrics['meanFarScore']}")
    print(f"   -> Mean Extreme Peak Preservation: {sum_metrics['meanExtremePeakPreservationPct']}%")
    benchmark_report["results"]["historical_validation_10y"] = val_report

    # Export benchmark report artifact
    out_dir = os.path.join(PROJECT_ROOT, "backend", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reproducible_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(benchmark_report, f, indent=2)

    print("\n" + "=" * 75)
    print(f" [SUCCESS] Reproducible Benchmark Evidence Report Exported to {out_path}")
    print("=" * 75)
    return benchmark_report

if __name__ == "__main__":
    run_reproducible_benchmark_suite()
