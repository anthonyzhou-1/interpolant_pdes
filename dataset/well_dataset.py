"""WellDataset that keeps HDF5 files open.

Upstream `WellDataset._load_one_sample` reopens the trajectory file through fsspec on
every `__getitem__`, which dominates step time on large splits. `FastWellDataset` reads
through a per-process cache of plain h5py handles instead; everything else is upstream's.
Handles are keyed by pid, so DataLoader workers never reuse a handle opened before fork.
"""

import os
from typing import Any, Dict

import h5py as h5
import numpy as np
from fsspec.implementations.local import LocalFileSystem
from the_well.data import WellDataset
from the_well.data.utils import IO_PARAMS, maximum_stride_for_initial_index


class FastWellDataset(WellDataset):
    """WellDataset with per-worker persistent HDF5 handles. See module docstring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handles: Dict[int, Any] = {}
        self._handle_pid = os.getpid()
        # h5py can only open local paths; remote stores fall back to fsspec.
        self._local = isinstance(self.fs, LocalFileSystem)

    def _file(self, file_idx: int):
        """Return an open h5py.File for `file_idx`, opening it in this process if needed."""
        pid = os.getpid()
        if pid != self._handle_pid:
            # Forked: drop the parent's handles without closing them.
            self._handles = {}
            self._handle_pid = pid
        handle = self._handles.get(file_idx)
        if handle is None:
            path = self.files_paths[file_idx]
            if self._local:
                handle = h5.File(path, "r")
            else:
                # Remote store (s3://, gcs://, ...): open via fsspec, once.
                handle = h5.File(
                    self.fs.open(path, "rb", **IO_PARAMS["fsspec_params"]),
                    "r",
                    **IO_PARAMS["h5py_params"],
                )
            self._handles[file_idx] = handle
        return handle

    def close(self):
        """Close any handles this process owns. Safe to call more than once."""
        if os.getpid() == self._handle_pid:
            for handle in self._handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
        self._handles = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        # Never pickle live HDF5 handles into a worker or a checkpoint.
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _load_one_sample(self, index):
        """Mirrors WellDataset._load_one_sample (the_well 1.2.0).

        The only change is the first line of the body: the `with h5.File(self.fs.open(...))`
        context manager is replaced by `self._file(file_idx)`, a handle that stays open. The
        rest is upstream's, kept in the same order so it can be diffed against it.
        """
        # Find specific file and local index
        if self.restriction_set is not None:
            index = self.restriction_set[index]
        file_idx = int(
            np.searchsorted(self.file_index_offsets, index, side="right") - 1
        )  # which file we are on
        windows_per_trajectory = self.n_windows_per_trajectory[file_idx]
        local_idx = index - max(
            self.file_index_offsets[file_idx], 0
        )  # First offset is -1
        sample_idx = local_idx // windows_per_trajectory
        time_idx = local_idx % windows_per_trajectory

        file = self._file(file_idx)  # persistent handle instead of a per-sample open

        # If we gave a stride range, decide the largest size we can use given the sample location
        dt = self.min_dt_stride
        if self.max_dt_stride > self.min_dt_stride:
            effective_max_dt = maximum_stride_for_initial_index(
                time_idx,
                self.n_steps_per_trajectory[file_idx],
                self.n_steps_input,
                self.n_steps_output,
            )
            effective_max_dt = min(effective_max_dt, self.max_dt_stride)
            if effective_max_dt > self.min_dt_stride:
                # Randint is non-inclusive on the upper bound
                dt = np.random.randint(self.min_dt_stride, effective_max_dt + 1)
        # Fetch the data
        data = {}

        output_steps = min(self.n_steps_output, self.max_rollout_steps)
        # If start_output_steps_at_t set, then work backwards for initial time index
        if self.full_trajectory_mode and self.start_output_steps_at_t >= 0:
            time_idx = self.start_output_steps_at_t - (self.n_steps_input) * dt

        data["variable_fields"], data["constant_fields"] = self._reconstruct_fields(
            file,
            self.caches[file_idx],
            sample_idx,
            time_idx,
            self.n_steps_input + output_steps,
            dt,
        )
        data["variable_scalars"], data["constant_scalars"] = self._reconstruct_scalars(
            file,
            self.caches[file_idx],
            sample_idx,
            time_idx,
            self.n_steps_input + output_steps,
            dt,
        )

        if self.boundary_return_type is not None:
            data["boundary_conditions"] = self._reconstruct_bcs(
                file,
                self.caches[file_idx],
                sample_idx,
                time_idx,
                self.n_steps_input + output_steps,
                dt,
            )

        if self.return_grid:
            data["space_grid"], data["time_grid"] = self._reconstruct_grids(
                file,
                self.caches[file_idx],
                sample_idx,
                time_idx,
                self.n_steps_input + output_steps,
                dt,
            )
        return data, file_idx, sample_idx, time_idx, dt
