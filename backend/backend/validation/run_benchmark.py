import os
import sys
import json
import csv
import numpy as np
from datetime import datetime, timezone

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data.era5_loader import ERA5DataLoader
from backend.data.nepsg_loader import load_nepsg_ensemble
from backend.historical_validation import HistoricalValidationEngine

def run_scientific_benchmark_suite():
    """
    Priority 7 — Reproducible Benchmark Evaluation Runner
    Generates:
    - outputs/validation/results.json
    - outputs/validation/results.csv
    
    For every metric stores:
    dataset, event, model, prediction, ground_truth, metric, value, timestamp, model_version, status
    """
    mode = os.getenv("STORMTRACE_MODE", "REAL")
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    model_version = "v2.0.0-"

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "validation")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "results.json")
    csv_path = os.path.join(out_dir, "results.csv")

    records = []

    print("==========================================================================")
    print(" StormTrace AI - Reproducible Scientific Benchmark Suite ")
    print(f" Mode: {mode} | Timestamp: {timestamp}")
    print("==========================================================================")

    # 1. ERA5 Dataset Ingestion Check
    era5_loader = ERA5DataLoader()
    era5_res = era5_loader.fetch_live_era5_dataset()
    era5_status = era5_res.get("status", "UNVERIFIED")

    records.append({
        "dataset": era5_res.get("source", "Copernicus ERA5"),
        "event": "India_Domain_Atmospheric_State",
        "model": "ERA5_Ingestion_Pipeline",
        "prediction": "Atmospheric_Grids",
        "ground_truth": "Copernicus_Reanalysis",
        "metric": "ERA5_Verification_Status",
        "value": era5_status,
        "timestamp": timestamp,
        "model_version": model_version,
        "status": era5_status
    })

    # 2. NEPS-G 50-Member Ensemble Verification
    neps_res = load_nepsg_ensemble()
    neps_status = neps_res.get("status", "UNVERIFIED")

    records.append({
        "dataset": neps_res.get("source", "NCMRWF NEPS-G"),
        "event": "50_Member_EPS_Forecast",
        "model": "EnsembleNWPEngine",
        "prediction": f"Ensemble_Members_Count_{neps_res.get('ensembleMembersCount', 0)}",
        "ground_truth": "NCMRWF_Operational_GRIB2",
        "metric": "NEPS_G_Verification_Status",
        "value": neps_status,
        "timestamp": timestamp,
        "model_version": model_version,
        "status": neps_status
    })

    # 3. 10-Year Historical Disaster Validation Evaluation
    validator = HistoricalValidationEngine()
    val_report = validator.evaluate_historical_case_studies()
    events_evaluated = val_report.get("caseStudies", [])

    for ev in events_evaluated:
        ev_id = ev.get("eventId", "unknown_event")
        ev_status = ev.get("status", "UNVERIFIED")
        contingency = ev.get("contingencyScores", {})
        tracking = ev.get("trackingValidation", {})
        downscaling = ev.get("downscalingPerformance", {})

        metrics_to_log = [
            ("CSI", contingency.get("csiScore", "N/A")),
            ("POD", contingency.get("podScore", "N/A")),
            ("FAR", contingency.get("farScore", "N/A")),
            ("Position_Error_KM", tracking.get("positionErrorKm", "N/A")),
            ("Peak_Preservation_Pct", downscaling.get("peakRainfallPreservationPct", "N/A"))
        ]

        for metric_name, val_str in metrics_to_log:
            records.append({
                "dataset": "Historical_Disaster_Ground_Truth",
                "event": ev_id,
                "model": "ST-GNN + DDPM Hybrid Pipeline",
                "prediction": "Model_Trajectory_And_Downscaled_Grid",
                "ground_truth": f"Historical_{ev_id}_Observation",
                "metric": metric_name,
                "value": str(val_str),
                "timestamp": timestamp,
                "model_version": model_version,
                "status": ev_status
            })

    # Save outputs/validation/results.json
    output_payload = {
        "timestamp": timestamp,
        "mode": mode,
        "model_version": model_version,
        "total_benchmark_records": len(records),
        "results": records
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Save outputs/validation/results.csv
    fieldnames = ["dataset", "event", "model", "prediction", "ground_truth", "metric", "value", "timestamp", "model_version", "status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    print(f"[SUCCESS] Benchmark execution completed. Records exported to:")
    print(f"   - {json_path}")
    print(f"   - {csv_path}")

    return output_payload

if __name__ == "__main__":
    run_scientific_benchmark_suite()
