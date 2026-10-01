"""CRPS / spread-skill ratio vs lead time for the climate models.

Rolls out --num_samples initial conditions, evenly spaced through the validation year,
with an --ensemble-member ensemble for --time_horizon 6 h steps, and scores every lead
time. Resumable per sample. Writes samples/<idx>.npz, results.npz and results.json
to --out/<slug>/. SSR follows the WeatherBench2 convention; ssr_corrected applies the
finite-ensemble factor sqrt((E+1)/E) (Fortin et al. 2014).
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
from common.loss import ensemble_crps, latitude_weight
from dataset.datamodule import PDEDataModule
from modules.train_module import TrainModule

from lightning.pytorch import seed_everything

# (name, variable, level index or None)
HEADLINE_FIELDS = [
    ("pr_6h",  "pr_6h", None),
    ("z500",   "zg",    7),
    ("t2m",    "tas",   None),
    ("u250",   "ua",    4),
    ("hus850", "hus",   10),
]
PER_SAMPLE_METRICS = ("crps", "skill", "spread", "mse", "var")


def score_sample(pred_ens, target, lat_w):
    '''
    pred_ens: dict of (1, E, T, nlat, nlon[, nlevel]); target: dict of (1, T, nlat, nlon[, nlevel]).
    Returns {name: {metric: (T,) float64 array}}.
    '''
    out = {}
    for name, var, lev in HEADLINE_FIELDS:
        p, y = pred_ens[var], target[var]
        if lev is not None:
            p, y = p[..., lev], y[..., lev]
        p, y = p.double(), y.double()
        crps, skill, spread = ensemble_crps(p, y, spatial_dims=(-2, -1), weight=lat_w,
                                            return_parts=True)
        mse = ((p.mean(1) - y) ** 2 * lat_w).mean(dim=(-2, -1))
        if p.shape[1] > 1:
            var_ = (p.var(dim=1, unbiased=True) * lat_w).mean(dim=(-2, -1))
        else:
            var_ = torch.zeros_like(mse)
        out[name] = {k: v[0].cpu().numpy() for k, v in
                     (("crps", crps), ("skill", skill), ("spread", spread),
                      ("mse", mse), ("var", var_))}
    return out


def aggregate(per_sample, ens):
    '''per_sample: {name: {metric: (n, T)}} -> {name: {curve: (T,)}}, NaN-robust over samples.'''
    agg = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, d in per_sample.items():
            mse, var_ = np.nanmean(d["mse"], 0), np.nanmean(d["var"], 0)
            ssr = np.sqrt(var_) / np.sqrt(mse)
            agg[name] = {"crps": np.nanmean(d["crps"], 0),
                         "crps_std": np.nanstd(d["crps"], 0),
                         "skill": np.nanmean(d["skill"], 0),
                         "spread": np.nanmean(d["spread"], 0),
                         "rmse": np.sqrt(mse),
                         "ssr": ssr,
                         "ssr_corrected": ssr * math.sqrt((ens + 1) / ens) if ens > 1 else ssr,
                         "n_finite": np.isfinite(d["crps"]).sum(0)}
    return agg


def main(args):
    torch.set_float32_matmul_precision("high")
    config = get_yaml(args.config)
    modelconfig, trainconfig, dataconfig = config["model"], config["training"], config["data"]
    modelconfig["model_name"] = args.model_name
    trainconfig["seed"] = args.seed
    trainconfig["plot_val"] = False
    dataconfig["batch_size"] = 1
    dataconfig["val_batch_size"] = 1
    dataconfig["dataset"]["val_nsteps"] = args.time_horizon
    dataconfig["num_val_batches"] = None
    dataconfig["num_val_samples"] = None
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
    sigma = model.sampler_sigma
    # deterministic samplers get a single member
    ens = args.ensemble if model.stochastic_sampling else 1
    if ens != args.ensemble:
        print(f"NOTE: {args.model_name}/{label} sampling is deterministic; using ensemble_size=1")

    var_tag = f"__{label}" if model.model_name in TrainModule.VARIANT_MODELS else ""
    if model.model_name == "interpolant" and math.isfinite(sigma):
        var_tag += f"_sigma{sigma:g}"
    slug = (f"climate__{args.model_name}{var_tag}__nfe{nfe}__E{ens}"
            f"__{args.time_horizon}steps__n{args.num_samples}__seed{args.seed}")
    out = os.path.join(args.out, slug)
    sample_dir = os.path.join(out, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    print(f"{args.model_name} / {label} (sigma={sigma}) at NFE {nfe}, E={ens}, "
          f"{args.time_horizon} steps, {args.num_samples} ICs from {args.checkpoint}\n -> {out}")

    if os.path.exists(os.path.join(out, "results.json")) and not args.overwrite:
        print("results.json already exists; pass --overwrite to redo. Nothing to do.")
        return

    # evenly spaced ICs, identical across models and seeds
    dataset = datamodule.val_dataset
    n_full = len(dataset)
    assert args.num_samples <= n_full, f"val split has only {n_full} rollouts of {args.time_horizon} steps"
    indices = np.linspace(0, n_full - 1, args.num_samples).round().astype(int).tolist()

    sample_path = lambda i: os.path.join(sample_dir, f"{i:05d}.npz")
    todo = [i for i in indices if args.overwrite or not os.path.exists(sample_path(i))]
    print(f"{len(indices) - len(todo)}/{len(indices)} samples already done")

    lat_w = latitude_weight(nlat, nlon, with_poles=dataconfig["with_poles"]).view(nlat, 1)
    lat_w = lat_w.to(device=device, dtype=torch.float64)

    loader = DataLoader(Subset(dataset, todo), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False,
                        prefetch_factor=2 if args.num_workers > 0 else None)
    with torch.no_grad():
        for batch, idx in zip(tqdm(loader, total=len(todo)), todo):
            # seeded per IC so resumed jobs are reproducible
            seed_everything(args.seed * 100_000 + idx, verbose=False)
            batch = [b.to(device, non_blocking=True) for b in batch]
            surface, multilevel, constants, yearly_constants, day, hour = batch
            _, _, target, _, pred_ens = model.predict_climate(
                surface, multilevel, day, hour, constants, yearly_constants,
                return_pred=True, ensemble_size=ens)
            scores = score_sample(pred_ens, target, lat_w)
            if not np.isfinite(scores["t2m"]["crps"]).all():
                bad = int(np.argmax(~np.isfinite(scores["t2m"]["crps"])))
                print(f"  WARNING: IC {idx} went non-finite at step {bad}")
            tmp = sample_path(idx) + ".tmp.npz"
            np.savez(tmp, **{f"{n}/{m}": v for n, d in scores.items() for m, v in d.items()})
            os.replace(tmp, sample_path(idx))
            del pred_ens, target

    # gather all samples, including those from earlier jobs
    per_sample = {n: {m: [] for m in PER_SAMPLE_METRICS} for n, _, _ in HEADLINE_FIELDS}
    for i in indices:
        with np.load(sample_path(i)) as f:
            for n in per_sample:
                for m in PER_SAMPLE_METRICS:
                    per_sample[n][m].append(f[f"{n}/{m}"])
    per_sample = {n: {m: np.stack(v) for m, v in d.items()} for n, d in per_sample.items()}
    agg = aggregate(per_sample, ens)

    lead_hours = 6 * np.arange(1, args.time_horizon + 1)
    np.savez(os.path.join(out, "results.npz"), lead_hours=lead_hours,
             ic_indices=np.array(indices),
             **{f"per_sample/{n}/{m}": v for n, d in per_sample.items() for m, v in d.items()},
             **{f"{n}/{c}": v for n, d in agg.items() for c, v in d.items()})

    meta = {"model_name": args.model_name, "sampler_variant": label,
            "sampler_sigma": sigma if math.isfinite(sigma) else None,
            "nfe": nfe, "ensemble_size": ens, "seed": args.seed,
            "time_horizon": args.time_horizon, "num_samples": len(indices),
            "ic_indices": indices, "checkpoint": args.checkpoint}
    curves = {n: {c: [None if not np.isfinite(x) else float(x) for x in v]
                  for c, v in d.items()} for n, d in agg.items()}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({**meta, "lead_hours": lead_hours.tolist(), "curves": curves}, f)

    for n, d in agg.items():
        pts = [(h, lead_hours.tolist().index(h)) for h in (24, 120, 240, 720) if h <= lead_hours[-1]]
        print(f"  {n:7s} " + "  ".join(f"{h // 24:>2d}d CRPS {d['crps'][k]:.4g} SSR {d['ssr_corrected'][k]:.3f}"
                                       for h, k in pts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRPS / SSR vs lead time of a climate model")
    parser.add_argument("--config", default="configs/climate.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="default",
                        help="edm / interpolant only: 'ode' (deterministic) or 'sde' (stochastic), "
                             "edm optionally suffixed with a solver, e.g. sde-euler")
    parser.add_argument("--sigma", type=float, default=None,
                        help="interpolant 'sde' only: sampling noise scale "
                             "(default: the sigma_coef it was trained with)")
    parser.add_argument("--nfe", type=int, default=10)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--time_horizon", type=int, default=120,
                        help="6 h steps: 40 = 10 days, 120 = 30 days")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="initial conditions, evenly spaced through the val year")
    parser.add_argument("--out", default="/path/to/data/logs/crps_results")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
