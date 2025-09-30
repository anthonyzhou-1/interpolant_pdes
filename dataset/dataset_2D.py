import h5py
import torch
from torch.utils.data import Dataset
import numpy as np 
from typing import Tuple

class PDEDataset2D(Dataset):
    """Load samples of a 2D PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 split: str,
                 resolution: list=None,
                 normalizer =  None,
                 return_traj = False,
                 horizon = None) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            split: [train, valid]
            resolution: resolution of the dataset [nt, nx, ny]
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.split = split
        self.return_traj = return_traj
        self.resolution = resolution
        self.nt, self.nx, self.ny = resolution
        self.horizon = horizon if horizon is not None else self.nt
        data = f[self.split]
        self.n_samples = len(data['u'])
        self.normalizer = normalizer

        self.u = data['u'] # (n_samples, nt, nx, ny) or (n_samples, nt, nx)
        self.x = torch.tensor(np.array(data['x'])) # (nx, ny, 2)
        self.t = torch.tensor(np.array(data['t'])) # (nt,)

        nt_data = self.t.shape[0] # original nt
        self.t_downsample = int(nt_data / self.nt)  # downsample factor 

        self.t = self.t[::self.t_downsample] # downsample time 
        self.t = self.t - self.t[0] # start time from zero
        self.t = self.t[:self.horizon] # truncate to horizon
        self.dt = self.t[1] - self.t[0]
        self.dx = self.x[1, 0, 0] - self.x[0, 0, 0]
        self.dy = self.x[0, 1, 1] - self.x[0, 0, 1]
        self.nt = len(self.t)

        print("Data loaded from: {}".format(path))
        print(f"dt: {self.dt:.3f}, dx: {self.dx:.3f}, nt: {self.nt}, nx: {self.nx}")
        print(f"Time ranges from {self.t[0]:.3f} to {self.t[-1]:.3f} = {self.dt:.3f} * {self.horizon} dt * nt")
        print("\n")

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, list]:
        """
        Get data item
        Args:
            i (int): data index
        Returns:
            u_input: torch.Tensor: input data at time t=0, shape [nx, ny, 1]
            u_label: torch.Tensor: label data from time t=1 to t=nt, shape [nt-1, nx, ny, 1]
            dx: float: spatial resolution
            dt: float: temporal resolution
            t: torch.Tensor: time in shape [nt]
        """

        u = torch.tensor(np.array(self.u[i]))
        u = u[::self.t_downsample].unsqueeze(-1) # truncate to nt by taking every t_downsample
        u = u[:self.horizon]
        u = self.normalizer.normalize(u) if self.normalizer is not None else u

        if not self.return_traj:
            rand_idx = np.random.randint(0, self.nt-1)
            u_input = u[rand_idx]
            u_label = u[rand_idx+1]
        else:
            u_input = u[0]
            u_label = u

        return_dict = {"input_fields": u_input, "output_fields": u_label, "dx": self.dx, "dt": self.dt, "t": self.t}
        return return_dict
