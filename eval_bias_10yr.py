"""Long autoregressive climate rollout for the climatological-bias table.

Rolls out one trajectory from the first frame of the PLASIM train split for
--time_horizon 6 h steps with true forcings, and scores the time-mean bias
(latitude-weighted RMSE) against the true trajectory and, if given, the reference
climatologies in --bias_path. Resumable via state.pt.

Outputs in --out/<slug>/: climatology.npz, global_mean.npz, results.json, and with
--save_rollout the full predicted state in rollout/ (see load_rollout()).
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common.utils import get_yaml
from common.loss import latitude_weighted_rmse
from dataset.datamodule import PDEDataModule
from dataset.plasim import SURFACE_FEATURES, MULTI_LEVEL_FEATURES
from modules.train_module import TrainModule

from lightning.pytorch import seed_everything

# (name in the table, variable, level index or None)
TABLE_FIELDS = [
    ("z500",   "zg",    7),
    ("t2m",    "tas",   None),
    ("t850",   "ta",    10),
    ("u250",   "ua",    4),
    ("pr_6h",  "pr_6h", None),
    ("hus850", "hus",   10),
]
# file names of the precomputed reference climatologies in --bias_path
REF_FILES = {"z500": "zg_50000.0", "t2m": "tas", "t850": "ta_85000.0",
             "u250": "ua_25000.0", "pr_6h": "pr_6h", "hus850": "hus_85000.0"}


def field(d, var, lev):
    return d[var] if lev is None else d[var][..., lev]


def lat_weights(nlat, nlon, device):
    # same latitude grid as latitude_weighted_rmse(with_poles=False)
    lat_end = (nlat - 1) * (360 / nlon) / 2
    lat = torch.linspace(-lat_end, lat_end, nlat, device=device, dtype=torch.float64)
    w = torch.cos(torch.deg2rad(lat))
    return (w / w.mean()).view(nlat, 1)


def bias_rmse(pred, target, nlat, nlon):
    # pred / target in shape nlat nlon
    return latitude_weighted_rmse(pred[None, None].double(), target[None, None].double(),
                                  with_poles=False, nlon=nlon, nlat=nlat).item()


def load_rollout(out_dir, start=0, stop=None):
    '''Steps [start, stop) of a saved rollout as (surface, multilevel) float32 arrays.'''
    d = os.path.join(out_dir, "rollout")
    meta = json.load(open(os.path.join(d, "meta.json")))
    stop = meta["n_steps"] if stop is None else min(stop, meta["n_steps"])
    blocks = sorted(int(f[len("surface_"):-4]) for f in os.listdir(d)
                    if f.startswith("surface_") and f.endswith(".npy"))
    parts = {"surface": [], "multilevel": []}
    for b in blocks:
        for kind in parts:
            a = np.load(os.path.join(d, f"{kind}_{b:06d}.npy"), mmap_mode="r")
            lo, hi = max(start, b), min(stop, b + len(a))
            if lo < hi:
                parts[kind].append(np.asarray(a[lo - b:hi - b]))
    surface, multilevel = (np.concatenate(parts[k]) for k in ("surface", "multilevel"))
    assert len(surface) == stop - start, "rollout blocks do not cover the requested steps"
    return surface, multilevel


def main(args):
    torch.set_float32_matmul_precision("high")
    config = get_yaml(args.config)
    modelconfig, trainconfig, dataconfig = config["model"], config["training"], config["data"]
    modelconfig["model_name"] = args.model_name
    trainconfig["seed"] = args.seed
    trainconfig["plot_val"] = False
    dataconfig["batch_size"] = 1
    dataconfig["dataset"]["training_nsteps"] = 1
    dataconfig["dataset"]["val_nsteps"] = 1
    nlat, nlon = dataconfig["nlat"], dataconfig["nlon"]

    seed_everything(args.seed)
    device = torch.device(args.device)

    datamodule = PDEDataModule(dataconfig)
    model = TrainModule(config=config, normalizer=datamodule.normalizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    label = model.set_sampler_variant(args.variant, sigma=args.sigma)
    nfe = model.set_nfe(args.nfe)
    model.set_ensemble(1)
    sigma = model.sampler_sigma

    var_tag = f"__{label}" if model.model_name in TrainModule.VARIANT_MODELS else ""
    if model.model_name == "interpolant":
        var_tag += f"_sigma{sigma:g}"
    slug = f"climate__{args.model_name}{var_tag}__nfe{nfe}__{args.time_horizon}steps__seed{args.seed}"
    out = os.path.join(args.out, slug)
    os.makedirs(out, exist_ok=True)
    print(f"{args.model_name} / {label} (sigma={sigma}) at NFE {nfe}, {args.time_horizon} steps "
          f"from {args.checkpoint}\n -> {out}")

    if os.path.exists(os.path.join(out, "results.json")) and not args.overwrite:
        print("results.json already exists; pass --overwrite to redo. Nothing to do.")
        return

    w = lat_weights(nlat, nlon, device)
    dataset = datamodule.train_dataset
    assert args.time_horizon <= len(dataset), f"train split has only {len(dataset)} steps"

    roll_dir = os.path.join(out, "rollout")
    if args.save_rollout:
        os.makedirs(roll_dir, exist_ok=True)
        with open(os.path.join(roll_dir, "meta.json"), "w") as f:
            json.dump({"n_steps": args.time_horizon, "dt_hours": 6,
                       "first_target_date": dataset.time_coords[1].strftime(),
                       "surface_vars": SURFACE_FEATURES, "multilevel_vars": MULTI_LEVEL_FEATURES,
                       "dtype": "float32", "units": "physical (as in the PLASIM h5)"},
                      f, indent=1)
    buf = {"surface": [], "multilevel": []}

    def flush_rollout(block_start):
        for kind, frames in buf.items():
            if frames:
                path = os.path.join(roll_dir, f"{kind}_{block_start:06d}.npy")
                with open(path + ".tmp", "wb") as f:
                    np.save(f, torch.stack(frames).numpy())
                os.replace(path + ".tmp", path)
            frames.clear()
    state_path = os.path.join(out, "state.pt")
    if os.path.exists(state_path) and not args.overwrite:
        state = torch.load(state_path, map_location=device, weights_only=False)
        start = state["i"]
        print(f"Resuming from step {start}")
        # reseed so the resumed segment does not replay earlier noise
        seed_everything(args.seed + start)
    else:
        state = {"i": 0, "z_pred": None, "pred_sum": None, "target_sum": None,
                 "gm_pred": {k: [] for k, _, _ in TABLE_FIELDS},
                 "gm_target": {k: [] for k, _, _ in TABLE_FIELDS}}
        start = 0

    loader = DataLoader(Subset(dataset, range(start, args.time_horizon)), batch_size=1,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True,
                        prefetch_factor=4 if args.num_workers > 0 else None)

    def save_state():
        tmp = state_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, state_path)

    z_pred = state["z_pred"]
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, initial=start, total=args.time_horizon), start=start):
            batch = [b.to(device, non_blocking=True) for b in batch]
            # after step 0 only forcings and targets are read from the batch
            _, pred, target, z_pred = model.validation_step(batch, batch_idx=i, eval=True,
                                                            z_pred=z_pred, ensemble_size=1)
            if args.save_rollout:
                buf["surface"].append(torch.stack(
                    [pred[k][0, 0] for k in SURFACE_FEATURES], dim=-1).float().cpu())
                buf["multilevel"].append(torch.stack(
                    [pred[k][0, 0] for k in MULTI_LEVEL_FEATURES], dim=-1).float().cpu())
            # every entry is b=1, t=1, nlat, nlon[, nlevel]
            pred = {k: v[0, 0].double() for k, v in pred.items()}
            target = {k: v[0, 0].double() for k, v in target.items()}
            if state["pred_sum"] is None:
                state["pred_sum"] = {k: torch.zeros_like(v) for k, v in pred.items()}
                state["target_sum"] = {k: torch.zeros_like(v) for k, v in target.items()}
            for k in pred:
                state["pred_sum"][k] += pred[k]
                state["target_sum"][k] += target[k]
            for name, var, lev in TABLE_FIELDS:
                state["gm_pred"][name].append((field(pred, var, lev) * w).mean().item())
                state["gm_target"][name].append((field(target, var, lev) * w).mean().item())

            if not math.isfinite(state["gm_pred"]["t2m"][-1]):
                raise RuntimeError(f"rollout went non-finite at step {i}")

            state["i"] = i + 1
            state["z_pred"] = z_pred
            if (i + 1) % args.save_every == 0:
                if args.save_rollout:
                    flush_rollout(i + 1 - len(buf["surface"]))
                save_state()
        if args.save_rollout:
            flush_rollout(state["i"] - len(buf["surface"]))

    n = state["i"]
    assert n == args.time_horizon, f"rollout stopped at {n} of {args.time_horizon} steps"
    clim_pred = {k: (v / n).cpu() for k, v in state["pred_sum"].items()}
    clim_target = {k: (v / n).cpu() for k, v in state["target_sum"].items()}

    np.savez(os.path.join(out, "climatology.npz"),
             **{f"pred/{k}": v.numpy() for k, v in clim_pred.items()},
             **{f"target/{k}": v.numpy() for k, v in clim_target.items()})
    dates = dataset.time_coords[1:n + 1]
    np.savez(os.path.join(out, "global_mean.npz"),
             year=np.array([d.year for d in dates]), month=np.array([d.month for d in dates]),
             day=np.array([d.day for d in dates]), hour=np.array([d.hour for d in dates]),
             **{f"pred/{k}": np.array(v) for k, v in state["gm_pred"].items()},
             **{f"target/{k}": np.array(v) for k, v in state["gm_target"].items()})

    bias_vs_truth, bias_vs_ref = {}, {}
    for name, var, lev in TABLE_FIELDS:
        p = field(clim_pred, var, lev)
        bias_vs_truth[name] = bias_rmse(p, field(clim_target, var, lev), nlat, nlon)
        if args.bias_path:
            ref = torch.from_numpy(np.load(os.path.join(args.bias_path, f"{REF_FILES[name]}_bias.npy")))
            bias_vs_ref[name] = bias_rmse(p, ref, nlat, nlon)

    results = {"model_name": args.model_name, "sampler_variant": label,
               "sampler_sigma": sigma if math.isfinite(sigma) else None,
               "nfe": nfe, "seed": args.seed, "time_horizon": n,
               "checkpoint": args.checkpoint,
               "bias_vs_truth": bias_vs_truth,
               "bias_vs_ref": bias_vs_ref or None, "bias_path": args.bias_path}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    if os.path.exists(state_path):
        os.remove(state_path)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10-year climatological bias of a climate model")
    parser.add_argument("--config", default="configs/climate.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="default",
                        help="edm / interpolant only: 'ode' (deterministic) or 'sde' (stochastic)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="interpolant 'sde' only: sampling noise scale "
                             "(default: the sigma_coef it was trained with)")
    parser.add_argument("--nfe", type=int, default=10)
    parser.add_argument("--time_horizon", type=int, default=14612,
                        help="6 h steps: 1460 = 1 yr, 14612 = 10 yr, 146095 = 100 yr")
    parser.add_argument("--bias_path", default=None,
                        help="directory holding the reference {var}_bias.npy climatologies")
    parser.add_argument("--out", default="/path/to/data/logs/bias_results")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_rollout", action="store_true",
                        help="also save the full predicted state at every step (~2.4 MB/step)")
    main(parser.parse_args())
