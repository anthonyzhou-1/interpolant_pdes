"""Evaluate a trained model across a range of NFEs, recording sampling wall-clock time.

    python eval_nfe.py --config=configs/km_flow.yaml --model_name=interpolant \
        --checkpoint=/path/to/model.ckpt --nfes 2 5 10 20 --out=/path/to/results
"""
import argparse
import csv
import json
import math
import os
import time
from datetime import datetime

import torch
import wandb

from common.utils import get_yaml, save_yaml
from dataset.datamodule import PDEDataModule
from modules.train_module import TrainModule

import lightning as L
from lightning.pytorch import seed_everything

DEFAULT_NFES = [1, 2, 5, 10, 20, 50, 100]

# columns written before the (dataset-dependent) val/* metrics
BASE_FIELDS = [
    "timestamp",
    "pde",
    "model_name",
    "seed",
    "dt_stride",
    "start_step",
    "checkpoint",
    "sampler_variant",
    "sampler_sigma",
    "ensemble_size",
    "requested_nfe",
    "effective_nfe",
    "num_val_batches",
    "num_val_samples",
    "n_val_samples",
    "wall_clock_s",
    "sample_time_s",
    "sample_calls",
    "sample_members",
    "sample_time_per_frame_ms",
    "sample_time_per_member_ms",
]


def process_args(args, config):
    modelconfig = config['model']
    trainconfig = config['training']
    dataconfig = config['data']

    if args.seed is not None:
        trainconfig["seed"] = args.seed
    if args.model_name is not None:
        modelconfig["model_name"] = args.model_name
    if args.dt_stride is not None:
        dataconfig['dataset']['dt_stride'] = args.dt_stride
    if args.start_step is not None:
        dataconfig['dataset']['start_output_steps_at_t'] = args.start_step

    # single device so timings are meaningful
    if trainconfig["accelerator"] == "gpu":
        trainconfig["devices"] = [int(args.device)]
    else:
        trainconfig["devices"] = 1  # cpu accelerator rejects a device index
    trainconfig["strategy"] = "auto"

    # None = full validation set; num_val_samples is batch-size independent
    dataconfig["num_val_batches"] = args.num_val_batches
    dataconfig["num_val_samples"] = args.num_val_samples

    # an ensemble of E rolls out B*E members, so large E needs a smaller batch
    if args.val_batch_size is not None:
        dataconfig["val_batch_size"] = args.val_batch_size
        dataconfig["batch_size"] = args.val_batch_size

    # skip plotting/dumps so they don't affect timing
    trainconfig["eval_all"] = False
    trainconfig["plot_val"] = False

    return config, modelconfig, trainconfig, dataconfig


def write_rows(path, rows):
    '''Atomically rewrite this job's result CSV.'''
    metric_fields = sorted({k for r in rows for k in r} - set(BASE_FIELDS))
    fields = BASE_FIELDS + metric_fields
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def jsonable(v):
    '''Map NaN/inf to None for valid JSON.'''
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def main(args):
    config = get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    pde = dataconfig['pde']
    model_name = modelconfig["model_name"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    description = args.description if args.description is not None else ""
    name = f"{model_name}_{pde}_{description}_{seed}_{now}_NFE"
    path = trainconfig["log_dir"] + name + "/"
    config['training']["log_dir"] = path
    os.makedirs(path, exist_ok=True)
    save_yaml(config, path + "config.yml")

    # result files are named by job identity, so a re-run overwrites its predecessor
    os.makedirs(args.out, exist_ok=True)
    tag = description or 'dt' + str(dataconfig['dataset'].get('dt_stride', ''))
    if args.start_step is not None:
        tag += f"_t{args.start_step}"
    var_tag = ("__" + "-".join(args.variants)
               if model_name in TrainModule.VARIANT_MODELS else "")
    slug = f"{pde}__{model_name}{var_tag}__{tag}__seed{seed}"
    csv_path = os.path.join(args.out, slug + ".csv")
    json_path = os.path.join(args.out, slug + ".json")

    run = wandb.init(project=args.wandb_project or (trainconfig["project"] + "_eval"),
                     name=name,
                     mode=args.wandb_mode or trainconfig.get("wandb_mode", "online"),
                     group=f"{pde}__{model_name}",
                     job_type="nfe_sweep",
                     tags=[pde, model_name, f"seed{seed}"],
                     config={"pde": pde, "model_name": model_name, "seed": seed,
                             "checkpoint": args.checkpoint,
                             "dt_stride": dataconfig['dataset'].get("dt_stride", ""),
                             "start_step": args.start_step,
                             "nfes": args.nfes, "ensemble": args.ensemble,
                             "variants": args.variants,
                             "num_val_batches": args.num_val_batches,
                             "num_val_samples": args.num_val_samples,
                             "val_batch_size": args.val_batch_size})
    wandb.define_metric("effective_nfe")
    wandb.define_metric("sweep/*", step_metric="effective_nfe")

    datamodule = PDEDataModule(dataconfig=dataconfig)
    model = TrainModule(config, normalizer=datamodule.normalizer)

    # load once and reuse across every NFE
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.time_sampling = True

    n_val_samples = len(datamodule.val_dataset)
    print(f"Evaluating {model_name} on {pde} ({n_val_samples} val samples) "
          f"from {args.checkpoint}")

    # edm and interpolant have deterministic (ode) and stochastic (sde) samplers
    variants = args.variants if model_name in TrainModule.VARIANT_MODELS else ["default"]
    requested_ensembles = sorted({max(1, int(e)) for e in args.ensemble})

    results = []
    for variant in variants:
        label = model.set_sampler_variant(variant, sigma=args.sigma)
        sigma = model.sampler_sigma
        if len(variants) > 1:
            print(f" sampler variant: {label}" + (f" (sigma={sigma})" if math.isfinite(sigma) else ""))

        # ensembles only make sense for a stochastic sampler
        ensembles = requested_ensembles
        if not model.stochastic_sampling and any(e > 1 for e in ensembles):
            why = ("interpolant with integrator='euler' is a deterministic ODE; use the 'sde' "
                   "variant (integrator='em')" if model_name == "interpolant"
                   else f"{model_name}/{label} sampling is deterministic")
            print(f"  NOTE: {why}, so ensemble members would be identical. "
                  f"Restricting to ensemble_size=1.")
            ensembles = [1]

        for ens in ensembles:
            model.set_ensemble(ens)
            if len(ensembles) > 1:
                print(f" ensemble size: {ens}")

            seen_nfes = set()
            for requested_nfe in args.nfes:
                effective_nfe = model.set_nfe(requested_nfe)

                # ddpm and deterministic baselines have a fixed NFE
                if effective_nfe in seen_nfes:
                    print(f"  skipping requested NFE {requested_nfe}: "
                          f"{model_name} is fixed at NFE {effective_nfe}")
                    continue
                seen_nfes.add(effective_nfe)

                trainer = L.Trainer(devices=trainconfig["devices"],
                                    accelerator=trainconfig["accelerator"],
                                    strategy=trainconfig["strategy"],
                                    log_every_n_steps=trainconfig["log_every_n_steps"],
                                    default_root_dir=path,
                                    logger=False,
                                    enable_checkpointing=False,
                                    num_sanity_val_steps=0)

                model.val_tag = f"{label}_nfe{effective_nfe}_E{ens}"

                model.reset_timing()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                metrics = trainer.validate(model=model, datamodule=datamodule)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                wall_clock_s = time.perf_counter() - t0

                metrics = metrics[0] if metrics else {}
                per_frame_ms = (1000.0 * model._sample_time_s / model._sample_calls
                                if model._sample_calls else float("nan"))
                per_member_ms = (1000.0 * model._sample_time_s / model._sample_members
                                 if model._sample_members else float("nan"))

                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "pde": pde,
                    "model_name": model_name,
                    "seed": seed,
                    "dt_stride": dataconfig['dataset'].get("dt_stride", ""),
                    "start_step": args.start_step if args.start_step is not None else "",
                    "checkpoint": args.checkpoint,
                    "sampler_variant": label,
                    "sampler_sigma": sigma if math.isfinite(sigma) else "",
                    "ensemble_size": ens,
                    "requested_nfe": requested_nfe,
                    "effective_nfe": effective_nfe,
                    "num_val_batches": args.num_val_batches if args.num_val_batches is not None else "full",
                    "num_val_samples": args.num_val_samples if args.num_val_samples is not None else "full",
                    "n_val_samples": n_val_samples,
                    "wall_clock_s": round(wall_clock_s, 3),
                    "sample_time_s": round(model._sample_time_s, 3),
                    "sample_calls": model._sample_calls,
                    "sample_members": model._sample_members,
                    "sample_time_per_frame_ms": round(per_frame_ms, 4),
                    "sample_time_per_member_ms": round(per_member_ms, 6),
                }
                row.update({k: v for k, v in metrics.items()})
                results.append(row)

                prefix = f"sweep/{label}/E{ens}"
                wandb.log({"effective_nfe": effective_nfe,
                           **{f"{prefix}/{k.replace('val/', '')}": v
                              for k, v in metrics.items()},
                           f"{prefix}/wall_clock_s": wall_clock_s,
                           f"{prefix}/ms_per_sampler_call": per_frame_ms,
                           f"{prefix}/ms_per_member": per_member_ms,
                           f"{prefix}/sampler_calls": model._sample_calls})

                # write after every point so partial sweeps are kept
                write_rows(csv_path, results)
                with open(json_path, "w") as f:
                    json.dump([{k: jsonable(v) for k, v in r.items()} for r in results],
                              f, indent=1)

                print(f"  NFE {effective_nfe:>4} (E={ens}): {wall_clock_s:8.1f}s wall, "
                      f"{per_frame_ms:7.2f} ms/step")

    if results:
        cols = BASE_FIELDS + sorted({k for r in results for k in r} - set(BASE_FIELDS))
        wandb.log({"nfe_sweep": wandb.Table(
            columns=cols, data=[[jsonable(r.get(c, None)) for c in cols] for r in results])})
        wandb.summary["n_sweep_points"] = len(results)

    wandb.finish()
    print(f"\nWrote {len(results)} row(s) to {csv_path} and {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model across a range of NFEs, recording wall-clock time')
    parser.add_argument("--config", default=None)
    parser.add_argument('--checkpoint', required=True, help='Path to the checkpoint to evaluate')
    parser.add_argument('--out', required=True,
                        help='Directory for this job\'s result CSV + JSON')
    parser.add_argument('--nfes', type=int, nargs='+', default=DEFAULT_NFES,
                        help='Sampling budgets to evaluate (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--device', type=int, default=0, help='Single GPU index to evaluate on')
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--description', default=None)
    parser.add_argument('--dt_stride', type=int, default=None,
                        help='Prediction lead time in raw timesteps (rayleigh_benard, km_flow)')
    parser.add_argument('--start_step', type=int, default=None,
                        help='Raw timestep of the first target frame (rayleigh_benard only)')
    parser.add_argument('--num_val_batches', type=int, default=None,
                        help='Cap validation at N batches (default: full val set)')
    parser.add_argument('--num_val_samples', type=int, default=None,
                        help='Cap validation at N samples; overrides --num_val_batches')
    parser.add_argument('--ensemble', type=int, nargs='+', default=[1],
                        help='Ensemble sizes to evaluate, e.g. --ensemble 1 8')
    parser.add_argument('--variants', nargs='+', default=['ode', 'sde'],
                        help="edm/interpolant only: 'ode' and/or 'sde' (edm may add a solver, "
                             "e.g. sde-heun)")
    parser.add_argument('--sigma', type=float, default=None,
                        help="interpolant only: noise scale for the 'sde' variant "
                             "(default: trained sigma_coef)")
    parser.add_argument('--val_batch_size', type=int, default=None,
                        help='Override the val batch size')
    parser.add_argument('--wandb_project', default=None,
                        help='Defaults to the training project with an _eval suffix')
    parser.add_argument('--wandb_mode', default=None, help='online / offline / disabled')
    args = parser.parse_args()

    main(args)
