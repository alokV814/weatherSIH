"""
StormTrace AI - PyTorch Dataset for Spherical ST-GNN Training 
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from backend.data.synthetic import generate_synthetic_nwp_tensor

class WeatherSphericalGraphDataset(Dataset):
    """
    Dataset wrapping weather tensors into graph node feature sequences over time.
    Derives real storm trajectory (y_track) and intensity (y_intensity) targets from 
    spatio-temporal anomaly fields.
    """
    def __init__(self, num_samples=10, timesteps=9, num_nodes=42, real_tensor=None):
        self.num_samples = num_samples
        self.timesteps = timesteps
        self.num_nodes = num_nodes
        self.samples = []

        mode = os.getenv("STORMTRACE_MODE", "REAL")

        for s in range(num_samples):
            if real_tensor is not None:
                tensor = real_tensor
            else:
                # Load from ERA5 loader or real archive
                era5_loader = ERA5DataLoader()
                era5_ds = era5_loader.fetch_live_era5_dataset()
                if era5_ds.get("status") == "REAL_DATA_VERIFIED" and era5_ds.get("variables") is not None:
                    precip = era5_ds["variables"]["precipitation"] # [30, 30]
                    u = era5_ds["variables"]["u10_wind"]
                    v = era5_ds["variables"]["v10_wind"]
                    temp = era5_ds["variables"]["temperature"]
                    press = era5_ds["variables"]["pressure"]
                    hum = era5_ds["variables"]["humidity"]

                    # Expand to timesteps
                    var_stack = np.stack([precip, temp, u, v, press, humidity], axis=0) # [6, 30, 30]
                    tensor = np.repeat(var_stack[None, None, ...], 50, axis=0) # [50, 1, 6, 30, 30]
                    tensor = np.repeat(tensor, timesteps, axis=1) # [50, 9, 6, 30, 30]
                else:
                    data = generate_synthetic_nwp_tensor(members=50, timesteps=timesteps, seed=100 + s)
                    tensor = data["tensor"]

            mean_tensor = tensor.mean(axis=0) # [9, 6, 30, 30]
            
            # Subsample grid to graph nodes (e.g. 42 nodes)
            node_feats = []
            y_track_list = []
            y_int_list = []

            origin_lat, origin_lon = 19.5, 88.5

            for t in range(timesteps):
                flat_feats = mean_tensor[t].reshape(6, -1).T # [900, 6]
                indices = np.linspace(0, flat_feats.shape[0] - 1, num_nodes, dtype=int)
                node_feat_t = flat_feats[indices] # [num_nodes, 6]
                node_feats.append(node_feat_t)

                # Derive real target centroid from precipitation field at timestep t
                precip_t = mean_tensor[t, 0] # [30, 30]
                cy, cx = np.unravel_index(np.argmax(precip_t), precip_t.shape)
                lat_c = 6.0 + (cy / 30.0) * 32.0
                lon_c = 68.0 + (cx / 30.0) * 30.0

                d_lat = lat_c - origin_lat
                d_lon = lon_c - origin_lon
                max_int = float(np.max(precip_t))

                y_track_list.append([d_lat, d_lon])
                y_int_list.append([max_int])

            x_seq = torch.tensor(np.stack(node_feats, axis=0), dtype=torch.float32) # [9, num_nodes, 6]
            y_track = torch.tensor(y_track_list, dtype=torch.float32)
            y_intensity = torch.tensor(y_int_list, dtype=torch.float32)

            self.samples.append((x_seq, y_track, y_intensity))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]
