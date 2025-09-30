import torch
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset
import pandas as pd
import h5py as h5f
import pickle 
import cftime

# defined on the surface of earth 
SURFACE_FEATURES = [
    'evap', # lwe_of_water_evaporation, NaN
    "mrro", # surface_runoff, NaN
    "mrso", # lwe_of_soil_moisture_content, NaN
    "pl", # log_surface_pressure
    "pr_12h", # 12-hour accumulated precipitation
    "pr_6h", # 6-hour accumulated precipitation
    "tas", # air_temperature_2m
    "ts", # surface_temperature, NaN 
]

# defined on many levels of the atmosphere
MULTI_LEVEL_FEATURES = [
    'hus', # specific_humidity
    "ta", # air_temperature
    "ua", # eastward_wind
    "va", # northward_wind
    "zg", # geopotential
]

# constant for all time
CONSTANTS_FEATURES = [
    'lsm', # land_binary_mask # nans
    'sg', # surface_geopotential
    'z0', # surface_roughness_length
]

# constant for each day, but repeat for each year
# Also defined for leap years. Has an extra day, which is 4 more intervals
YEARLY_FEATURES = [
    'rsdt', # TOA (Top of Atmosphere) Incident Shortwave Radiation 
    'sic', # sea_ice_cover # nans
    'sst', # surface_temperature # nans
]

class Normalizer:
    def __init__(self, stat_path, features_names=SURFACE_FEATURES + MULTI_LEVEL_FEATURES, ae=False):
        # stat_dict: {feature_name: (mean, std)}
        self.features_names = features_names
        surface_stats, multilevel_stats = self.load_norm_stats(stat_path)
        self.surface_stats = surface_stats  # shape (surface_channels, 2)
        self.multilevel_stats = multilevel_stats # shape (nlevels, multi_level_channels, 2)

        self.surface_means = torch.tensor(surface_stats[:, 0], dtype=torch.float32) # shape (surface_channels)
        self.surface_means = rearrange(self.surface_means, 'c -> 1 1 1 c') 

        self.surface_stds = torch.tensor(surface_stats[:, 1], dtype=torch.float32) # shape (surface_channels)
        self.surface_stds = rearrange(self.surface_stds, 'c -> 1 1 1 c') 

        self.multilevel_means = torch.tensor(multilevel_stats[:, :, 0], dtype=torch.float32) # shape (nlevels, multi_level_channels)
        self.multilevel_means = rearrange(self.multilevel_means, 'n c -> 1 1 1 n c')
        self.multilevel_stds = torch.tensor(multilevel_stats[:, :, 1], dtype=torch.float32) # shape (nlevels, multi_level_channels) 
        self.multilevel_stds = rearrange(self.multilevel_stds, 'n c -> 1 1 1 n c')

        if ae:
            self.surface_means = self.surface_means.squeeze(0) # 1 1 c 
            self.surface_stds = self.surface_stds.squeeze(0)
            self.multilevel_means = self.multilevel_means.squeeze(0)
            self.multilevel_stds = self.multilevel_stds.squeeze(0)

        self.surface_nans = [0, 1, 2, 7]

    def load_norm_stats(self, norm_stat_path):
        surface_stats = []
        multilevel_stats = []
        with np.load(norm_stat_path, allow_pickle=True) as f:
            normalize_mean, normalize_std = f['normalize_mean'].item(), f['normalize_std'].item()
            for feature_name in self.features_names:
                assert feature_name in normalize_mean.keys(), f'{feature_name} not in {norm_stat_path}'
                mean = normalize_mean[feature_name] 
                std = normalize_std[feature_name]

                if feature_name in SURFACE_FEATURES:
                    surface_stats.append([mean, std])
                elif feature_name in MULTI_LEVEL_FEATURES:
                    multilevel_stats.append([mean, std])
        # shape (8, 2), (13, 5, 2)
        return np.array(surface_stats), np.array(multilevel_stats).transpose(2, 0, 1)
    
    def normalize(self, surface_feat, multilevel_feat):
        # surface feat in shape (nt, nlat, nlon, surface_channels)
        # multilevel feat in shape (nt, nlat, nlon, nlevel, multi_level_channels)
        # assume this runs on cpu threads for dataloader

        for nan_idx in self.surface_nans:
            # replace nan w/ mean of the feature
            surface_feat[..., nan_idx] = torch.nan_to_num(surface_feat[..., nan_idx], nan=self.surface_means[..., nan_idx].item())

        surface_feat = (surface_feat - self.surface_means) / self.surface_stds
        multilevel_feat = (multilevel_feat - self.multilevel_means) / self.multilevel_stds

        return surface_feat, multilevel_feat

    def denormalize(self, surface_feat, multilevel_feat):
        # surface feat in shape (nt, nlat, nlon, surface_channels)
        # multilevel feat in shape (nt, nlat, nlon, nlevel, multi_level_channels)

        surface_feat = surface_feat * self.surface_stds.to(surface_feat.device) + self.surface_means.to(surface_feat.device)
        multilevel_feat = multilevel_feat * self.multilevel_stds.to(surface_feat.device) + self.multilevel_means.to(surface_feat.device)

        return surface_feat, multilevel_feat
    
    def batch_normalize(self, surface_feat, multilevel_feat):
        # surface feat in shape (b, nt, nlat, nlon, surface_channels)
        # multilevel feat in shape (b, nt, nlat, nlon, nlevel, multi_level_channels)     

        surface_feat = (surface_feat - self.surface_means.unsqueeze(0).to(surface_feat.device)) / self.surface_stds.unsqueeze(0).to(surface_feat.device)
        multilevel_feat = (multilevel_feat - self.multilevel_means.unsqueeze(0).to(surface_feat.device)) / self.multilevel_stds.unsqueeze(0).to(surface_feat.device)

        return surface_feat, multilevel_feat  

    def batch_denormalize(self, surface_feat, multilevel_feat):
        # surface feat in shape (b, nt, nlat, nlon, surface_channels)
        # multilevel feat in shape (b, nt, nlat, nlon, nlevel, multi_level_channels)

        surface_feat = surface_feat * self.surface_stds.unsqueeze(0).to(surface_feat.device) + self.surface_means.unsqueeze(0).to(surface_feat.device)
        multilevel_feat = multilevel_feat * self.multilevel_stds.unsqueeze(0).to(surface_feat.device) + self.multilevel_means.unsqueeze(0).to(surface_feat.device)

        return surface_feat, multilevel_feat
        
class PLASIMData(Dataset):
    def __init__(self,
                 data_path,
                 norm_stats_path,
                 boundary_path,
                 time_path,
                 split="train",
                 surface_vars=SURFACE_FEATURES,
                 multi_level_vars=MULTI_LEVEL_FEATURES,
                 constant_names=CONSTANTS_FEATURES,
                 yearly_names=YEARLY_FEATURES,
                 normalize_feature=True,
                 interval=1,  # interval=1 is equal to 6 hours
                 nsteps=2,   # spit out how many consecutive future sequences
                 load_into_memory=False,
                 output_timecoords=False,
                 ae=False,
                 ):

        self.data_path = data_path  # a zarr file
        self.file = h5f.File(self.data_path, 'r') # has keys of 'split'
        self.data = self.file[split] # has keys of 'surface', 'multilevel', 'lat', 'lon', 'hour', 'day'
        self.features_names = surface_vars + multi_level_vars
        self.constant_names = constant_names
        self.yearly_names = yearly_names
        self.normalize_feature = normalize_feature
        self.output_timecoords = output_timecoords
        self.ae = ae

        # this assumes that the normalization statistics are stored in the same directory as the data
        self.normalizer = Normalizer(norm_stats_path, self.features_names, ae = ae)

        self.interval = interval
        self.nsteps = nsteps

        # load the constants, in shape (nlat, nlon, nconstants) or (ntime, nlat, nlon, nyearly)
        self.constants, self.yearly_constants, self.leap_yearly_constants = self.load_constants(boundary_path)

        self.surface_vars = surface_vars
        self.multi_level_vars = multi_level_vars

        # get the time stamps
        with open(time_path, 'rb') as f:
            self.time_coords = pickle.load(f) # array of cftime objects
            self.time_coords = self.time_coords.values

        # filter out those will be out of bound
        if nsteps > 0:
            self.time_coords_filtered = self.time_coords[:-(interval * nsteps)]
        else:
            self.time_coords_filtered = self.time_coords

        if load_into_memory:
            self.surface = torch.from_numpy(self.data['surface'][:])
            self.multilevel = torch.from_numpy(self.data['multilevel'][:])
        else:
            self.surface = self.data['surface'] # t nlat nlon nsurface_channels
            self.multilevel = self.data['multilevel'] # t nlat nlon nlevels nmulti_channels

        self.hour = torch.from_numpy(self.data['hour'][:])
        self.day = torch.from_numpy(self.data['day'][:])
        self.load_into_memory = load_into_memory

        self.nstamps = len(self.time_coords_filtered)
        print(f"Loaded {self.nstamps} time stamps for {split} split, from {self.time_coords[0].strftime()} to {self.time_coords[-1].strftime()}")
        print(f"Normalize: {self.normalize_feature}, Load into memory: {self.load_into_memory}")

    def __len__(self):
        return self.nstamps

    def get_var_names(self):
        return self.surface_vars, self.multi_level_vars, self.constant_names, self.yearly_names
    
    def load_constants(self, boundary_path):
        # load constants into local memory. About 300 Mb
        boundary_file = h5f.File(boundary_path, 'r')
        constants_dict = {}
        for constant in self.constant_names:
            assert constant in boundary_file.keys(), f'{constant} not in the boundary file'
            constant_values = boundary_file[constant][:] # (nlat, nlon) for constants, (time, nlat, lon) for yearly
            # scale to [-1, 1]
            scaled_constant = 2 * (constant_values - constant_values.min()) / (constant_values.max() - constant_values.min()) - 1
            constants_dict[constant] = torch.from_numpy(scaled_constant).float()

        yearly_constants_dict = {}
        for constant in self.yearly_names:
            assert constant in boundary_file.keys(), f'{constant} not in the boundary file'
            constant_values = boundary_file[constant][:] # (nlat, nlon) for constants, (time, nlat, lon) for yearly
            # scale to [-1, 1]
            scaled_constant = 2 * (constant_values - constant_values.min()) / (constant_values.max() - constant_values.min()) - 1
            yearly_constants_dict[constant] = torch.from_numpy(scaled_constant).float()

        leap_yearly_names = [f'{constant}_leap' for constant in self.yearly_names]
        leap_yearly_constants_dict = {}
        for constant in leap_yearly_names:
            assert constant in boundary_file.keys(), f'{constant} not in the boundary file'
            constant_values = boundary_file[constant][:] # (nlat, nlon) for constants, (time, nlat, lon) for yearly
            # scale to [-1, 1]
            scaled_constant = 2 * (constant_values - constant_values.min()) / (constant_values.max() - constant_values.min()) - 1
            leap_yearly_constants_dict[constant] = torch.from_numpy(scaled_constant).float()
        boundary_file.close()

        constants = torch.stack([constants_dict[k] for k in self.constant_names], dim=-1) # (nlat, nlon, nconstants)
        yearly_constants = torch.stack([yearly_constants_dict[k] for k in self.yearly_names], dim=-1) # (ntime, nlat, nlon, nyearly)
        leap_yearly_constants = torch.stack([leap_yearly_constants_dict[k] for k in leap_yearly_names], dim=-1) # (ntime_leap, nlat, nlon, nyearly)
        return constants, yearly_constants, leap_yearly_constants
    
    def __getitem__(self, idx):
        # fetch time coord first
        start_time_idx = idx
        end_time_idx =  start_time_idx + self.interval * (self.nsteps + 1)
        idx_range = range(start_time_idx, end_time_idx, self.interval)
        time_coord = self.time_coords[idx_range]

        surface_feat = self.surface[idx_range] # (nsteps+1, nlat, nlon, nsurface_channels)
        multilevel_feat = self.multilevel[idx_range] # (nsteps+1, nlat, nlon, nlevels, nmulti_channels)
        day_of_year = self.day[idx_range] # (nsteps+1)
        hour_of_day = self.hour[idx_range]

        if not self.load_into_memory:
            surface_feat = torch.from_numpy(surface_feat)
            multilevel_feat = torch.from_numpy(multilevel_feat)

        if self.nsteps == 0:
            surface_feat = surface_feat.squeeze() # (nlat, nlon, nsurface_channels)
            multilevel_feat = multilevel_feat.squeeze() # (nlat, nlon, nlevels, nmulti_channels)

        # normalize the feature
        if self.normalize_feature:
            surface_feat, multilevel_feat = self.normalizer.normalize(surface_feat, multilevel_feat)

        # get temporal coords
        timestamp = [pd.Timestamp(t.strftime()) for t in time_coord]

        # check if all time coords are during a leap year
        leap_years = [cftime.is_leap_year(time_coord_i.year, 'proleptic_gregorian') for time_coord_i in time_coord]
        
        yearly_constants = []
        for i in range(len(time_coord)):
            year_string = time_coord[i].strftime('%Y')
            num_hours = cftime.date2num(time_coord[i], f'hours since {year_string}-01-01 00:00:00', calendar='proleptic_gregorian')
            yearly_idx = int(num_hours // 6) # 6 hours per interval
            if leap_years[i]:
                yearly_constants_i = self.leap_yearly_constants[yearly_idx] # (nlat, nlon, nyearly)
            else:
                yearly_constants_i = self.yearly_constants[yearly_idx] # (nlat, nlon, nyearly)
            yearly_constants.append(yearly_constants_i)
        
        yearly_constants = torch.stack(yearly_constants, dim=0) # (nt, nlat, nlon, nyearly)

        if not self.output_timecoords:
            return surface_feat, multilevel_feat,\
                   self.constants.clone(), yearly_constants, day_of_year, hour_of_day
        else:
            return surface_feat, multilevel_feat,\
                   self.constants.clone(), yearly_constants, day_of_year, hour_of_day, timestamp











