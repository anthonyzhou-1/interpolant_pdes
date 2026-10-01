import lightning as L
import torch
from torch.utils.data import DataLoader, Subset
from dataset.dataset_2D import PDEDataset2D
from dataset.well_dataset import FastWellDataset
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
        # Val samples are full trajectories (much larger than train samples), so the
        # val loader can be sized separately; defaults to the train values.
        self.val_batch_size = dataconfig.get("val_batch_size", self.batch_size)
        self.val_num_workers = dataconfig.get("val_num_workers", self.num_workers)
        self.prefetch_factor = dataconfig.get("prefetch_factor", 2)  # torch's default
        # Avoid re-spawning workers (and reopening HDF5 files) after every validation.
        self.persistent_workers = dataconfig.get("persistent_workers", True)
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
                                                horizon=self.dataset_config.get("horizon", None),
                                                dt_stride=self.dataset_config.get("dt_stride", None),)
            self.val_dataset = PDEDataset2D(path = self.dataset_config["valid_path"],
                                            split = "valid",
                                            resolution = self.dataset_config["resolution"],
                                            normalizer = self.normalizer,
                                            return_traj=False if self.ae else True,
                                            horizon=self.dataset_config.get("horizon", None),
                                            dt_stride=self.dataset_config.get("dt_stride", None),)
            
        elif self.pde == "rayleigh_benard": 
            from the_well.data.normalization import (
                ZScoreNormalization,
            )
            base_path = self.dataset_config["base_path"]
            # prediction lead time: number of raw Well timesteps between input and target frame
            dt_stride = self.dataset_config.get("dt_stride", 2)
            # grids and boundary masks are unused, so skip loading them
            self.train_dataset = FastWellDataset(
                well_base_path=f"{base_path}/datasets",
                well_dataset_name=self.pde,
                well_split_name="train",
                n_steps_input=1,
                n_steps_output=1,
                use_normalization=True,
                normalization_type = ZScoreNormalization,
                normalization_path=self.normalizer_config["stat_path"],
                min_dt_stride=dt_stride, 
                max_dt_stride=dt_stride,
                return_grid=False,
                boundary_return_type=None,
            )
            
            self.val_dataset = FastWellDataset(
                well_base_path=f"{base_path}/datasets",
                well_dataset_name=self.pde,
                well_split_name="valid",
                n_steps_input=1,
                n_steps_output=1,
                use_normalization=True,
                full_trajectory_mode = False if self.ae else True,
                # raw Well step of the first target frame (-1 = right after the input)
                start_output_steps_at_t = -1 if self.ae else self.dataset_config.get("start_output_steps_at_t", -1),
                normalization_type = ZScoreNormalization,
                normalization_path=self.normalizer_config["stat_path"],
                min_dt_stride=dt_stride, # take every dt_stride steps
                max_dt_stride=dt_stride,
                return_grid=False,
                boundary_return_type=None,
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

        self._subset_val_dataset()

    def _subset_val_dataset(self):
        '''
        Restrict validation to a fixed random subset of the val split.

        Sized either as `num_val_samples` (an absolute count) or as `num_val_batches`
        batches; the absolute count wins where both are set.

        Prefer num_val_samples whenever runs with different batch sizes have to be
        compared. num_val_batches multiplies by val_batch_size, so the same setting
        means different amounts of data to an ensemble job (which shrinks the batch to
        fit B*E members) than to a deterministic one -- and models scored on different
        amounts of data are not comparable. An absolute count is batch-size independent,
        and because the subset is a fixed permutation truncated at k, every job that asks
        for the same k evaluates exactly the same samples.

        A seeded random subset rather than Lightning's limit_val_batches, which would take
        the first N batches -- the Well val split is ordered by file, so those would all share
        one Rayleigh number. The seed is deliberately independent of training.seed so that every
        seed and every model validates on exactly the same data.
        '''
        num_val_samples = self.data_config.get("num_val_samples", None)
        num_val_batches = self.data_config.get("num_val_batches", None)
        if num_val_samples is None and num_val_batches is None:
            return

        n_full = len(self.val_dataset)
        if num_val_samples is not None:
            k = min(int(num_val_samples), n_full)
            sized_as = f"num_val_samples={num_val_samples}"
        else:
            k = min(int(num_val_batches) * self.val_batch_size, n_full)
            sized_as = f"num_val_batches={num_val_batches}"
        if k >= n_full:
            print(f"[PDEDataModule] {sized_as} covers the full val set "
                  f"({n_full} samples); not subsetting.")
            return

        generator = torch.Generator().manual_seed(self.data_config.get("val_subset_seed", 0))
        indices = torch.randperm(n_full, generator=generator)[:k].tolist()
        indices = sorted(indices)  # keep reads roughly sequential on HDF5/Well

        self.full_val_dataset = self.val_dataset
        self.val_dataset = Subset(self.val_dataset, indices)
        print(f"[PDEDataModule] validating on {k}/{n_full} samples "
              f"({sized_as}, {k / self.val_batch_size:.3g} batches of "
              f"{self.val_batch_size}, val_subset_seed="
              f"{self.data_config.get('val_subset_seed', 0)})")

    def prepare_data(self):
        pass
        
    def setup(self, stage: str):
        if stage == "fit":
            pass 

        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self, shuffle=True):
        self.pin_memory = False if self.num_workers == 0 else True
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=shuffle, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          **kwargs)

    def val_dataloader(self):
        # Leaner than the train loader (no pinned memory) since val samples are large.
        kwargs = {}
        if self.val_num_workers > 0:
            kwargs["prefetch_factor"] = 2
            kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(self.val_dataset, 
                          batch_size=self.val_batch_size, 
                          shuffle=False, 
                          num_workers=self.val_num_workers,
                          pin_memory=False,
                          **kwargs)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None
