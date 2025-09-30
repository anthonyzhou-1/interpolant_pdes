import lightning as L
from torch.utils.data import DataLoader
from dataset.dataset_2D import PDEDataset2D
from dataset.normalizer import ScalarNormalizer, WellNormalizer
import os 

class PDEDataModule(L.LightningDataModule):
    def __init__(self, 
                 dataconfig,) -> None:
        
        super().__init__()
        self.data_config = dataconfig
        self.dataset_config = dataconfig["dataset"]
        self.batch_size = dataconfig["batch_size"]
        self.num_workers = dataconfig["num_workers"]
        self.pde = dataconfig['pde']
        self.normalizer_config = dataconfig["normalizer"]
        self.ae = dataconfig.get("ae", False)

        if self.pde == "km_flow": 
            if not os.path.exists(self.normalizer_config["stat_path"]):
                # generate normalization statistics
                self.normalizer = ScalarNormalizer(stat_path=self.normalizer_config["stat_path"],
                                                dataset=PDEDataset2D(path = self.dataset_config["train_path"],
                                                                    split = "train",
                                                                    resolution = self.dataset_config["resolution"],
                                                                    return_traj=True))
            else:
                self.normalizer = ScalarNormalizer(stat_path=self.normalizer_config["stat_path"])

            self.train_dataset = PDEDataset2D(path = self.dataset_config["train_path"],
                                                split = "train",
                                                resolution = self.dataset_config["resolution"],
                                                normalizer = self.normalizer,
                                                horizon=self.dataset_config.get("horizon", None),)
            self.val_dataset = PDEDataset2D(path = self.dataset_config["valid_path"],
                                            split = "valid",
                                            resolution = self.dataset_config["resolution"],
                                            normalizer = self.normalizer,
                                            return_traj=False if self.ae else True,
                                            horizon=self.dataset_config.get("horizon", None),)
            
        elif self.pde == "rayleigh_benard": 
            from the_well.data import WellDataset
            from the_well.data.normalization import (
                ZScoreNormalization,
            )
            base_path = self.dataset_config["base_path"]
            self.train_dataset = WellDataset(
                well_base_path=f"{base_path}/datasets",
                well_dataset_name=self.pde,
                well_split_name="train",
                n_steps_input=1,
                n_steps_output=1,
                use_normalization=True,
                normalization_type = ZScoreNormalization,
                normalization_path=self.normalizer_config["stat_path"],
                min_dt_stride=2, 
                max_dt_stride=2,
            )
            
            self.val_dataset = WellDataset(
                well_base_path=f"{base_path}/datasets",
                well_dataset_name=self.pde,
                well_split_name="valid",
                n_steps_input=1,
                n_steps_output=1,
                use_normalization=True,
                full_trajectory_mode = False if self.ae else True,
                normalization_type = ZScoreNormalization,
                normalization_path=self.normalizer_config["stat_path"],
                min_dt_stride=2, # take every 2 steps
                max_dt_stride=2,
            )

            self.normalizer = WellNormalizer(self.train_dataset.norm)

        elif self.pde == "climate":
            from dataset.plasim import PLASIMData
            self.train_dataset = PLASIMData(data_path=self.dataset_config["train_data_path"],
                                            norm_stats_path=self.normalizer_config["norm_stats_path"],
                                            boundary_path=self.dataset_config["boundary_path"],
                                            time_path=self.dataset_config["train_times_path"],
                                            nsteps=self.dataset_config["training_nsteps"],   
                                            normalize_feature=True,
                                            ae = dataconfig["ae"],
                                            split='train')
            
            self.val_dataset = PLASIMData(data_path=self.dataset_config["val_data_path"],
                                            norm_stats_path=self.normalizer_config["norm_stats_path"],
                                            boundary_path=self.dataset_config["boundary_path"],
                                            time_path=self.dataset_config["val_times_path"],
                                            nsteps=self.dataset_config["val_nsteps"],   
                                            normalize_feature=True,
                                            ae = dataconfig["ae"],
                                            split="valid")
        
            self.normalizer = self.val_dataset.normalizer

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # Eager imports to avoid specific dependencies that are not needed in most cases

        if stage == "fit":
            pass 

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self, shuffle=True):
        self.pin_memory = False if self.num_workers == 0 else True
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=shuffle, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None
