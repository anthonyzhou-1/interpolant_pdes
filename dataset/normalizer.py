from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os.path
import torch 
import numpy as np
from tqdm import tqdm

class WellNormalizer:
    # small wrapper to implement denormaliztion for z-score normalizer, since it's not in the repo
    def __init__(self,
                 well_normalizer,):
        self.well_normalizer = well_normalizer

    def denormalize(self, x):
        return self.well_normalizer.denormalize_flattened(x, mode="variable")

class ScalarNormalizer:
    def __init__(self,
                 stat_path = "./",
                 dataset=None,
                 scaler = "normal",
                 recalculate = False,
                 scaling_factor=1):
        self.scaler = scaler # normal or minmax
        self.scaling_factor = scaling_factor

        if os.path.isfile(stat_path) and not recalculate:
            self.load_stats(stat_path)
            print("Statistics loaded from", stat_path)
        else:
            assert dataset is not None, "Data must be provided for normalization"
            print("Calculating statistics for normalization")
            dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers=0)
            
            if self.scaler == "normal":
                u_scaler = StandardScaler()
            elif self.scaler == "minmax":
                u_scaler = MinMaxScaler(feature_range=(-1, 1)) 
            else:
                raise ValueError("Scaler must be either 'normal' or 'minmax'")

            for batch in tqdm(dataloader):
                u = batch["u_label"]
                u_scaler.partial_fit(u.reshape(-1, 1))

            if self.scaler == "normal":
                self.u_mean = u_scaler.mean_.item()
                self.u_std = np.sqrt(u_scaler.var_).item()
            else:
                self.u_min = u_scaler.min_.item()
                self.u_scale = u_scaler.scale_.item()
                
            self.save_stats(path=stat_path)
            print("Statistics saved to", stat_path)

        self.print_stats()

        if self.scaler == "normal":
            self.u_mean = torch.tensor(self.u_mean)
            self.u_std = torch.tensor(self.u_std)
        else:
            self.u_min = torch.tensor(self.u_min)
            self.u_scale = torch.tensor(self.u_scale)

    def print_stats(self):
        if self.scaler == "minmax":
            print(f"u min: {self.u_min}, u scale: {self.u_scale}")
            print(f"scaling factor: {self.scaling_factor}") 
        else:
            print(f"u mean: {self.u_mean}, u std: {self.u_std}")
            print(f"scaling factor: {self.scaling_factor}")

    def save_stats(self, path):
        if self.scaler == "minmax":
            with open(path, "wb") as f:
                pickle.dump([self.u_min, self.u_scale], f)
        else:
            with open(path, "wb") as f:
                pickle.dump([self.u_mean, self.u_std], f)

    def load_stats(self, path):
        with open(path, "rb") as f:
            if self.scaler == "minmax":
                self.u_min, self.u_scale = pickle.load(f)
            else:
                self.u_mean, self.u_std = pickle.load(f)
    
    def normalize(self, u):
        u_norm = u.clone()
        if self.scaler == "normal":
            u_norm = (u_norm - self.u_mean) / self.u_std
        else:
            u_norm = u_norm * self.u_scale + self.u_min

        u_norm = u_norm * self.scaling_factor
        return u_norm

    def denormalize(self, u): 
        u_denorm = u.clone()
        u_denorm = u_denorm / self.scaling_factor

        if self.scaler == "normal":
            u_denorm = u_denorm * self.u_std + self.u_mean
        else:
            u_denorm = (u_denorm - self.u_min) / self.u_scale

        return u_denorm