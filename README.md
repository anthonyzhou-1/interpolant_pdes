# Reframing Generative Models for Physical Systems using Stochastic Interpolants
Anthony Zhou, Alexander Wikner, Amaury Lancelin, Pedram Hassanzadeh, Amir Barati Farimani [Paper](https://arxiv.org/abs/2509.26282)

## Requirements
```
conda create -n "my_env"
python -m pip install lightning
pip install wandb h5py einops scikit-learn tqdm scipy matplotlib pandas cftime xarray zarr
pip install git+https://github.com/KellerJordan/Muon   # Muon optimizer (training default)
pip install the_well                                   # Rayleigh-Benard
pip install torch-harmonics                            # SFNO baseline
```
To evaluate CRPS/SSR with [WeatherBench2](https://weatherbench2.readthedocs.io/en/latest/):
```
git clone git@github.com:google-research/weatherbench2.git
cd weatherbench2 && pip install .
```

## Datasets
- **Kolmogorov Flow**: generated with [ApeBench](https://github.com/tum-pbs/apebench); the data used here is from this [paper](https://www.sciencedirect.com/science/article/pii/S0045782525002622) and is on [Huggingface](https://huggingface.co/datasets/ayz2/temporal_pdes).
- **Rayleigh-Benard**: from [The Well](https://github.com/PolymathicAI/the_well) ([dataset page](https://polymathic-ai.org/the_well/datasets/rayleigh_benard/)).
- **Climate (PlaSim)**: will be released in another publication.

Set the dataset, normalization-stat, and log paths (`/path/to/data/...`) in `configs/`.

## Training
Autoencoder:
```
python train.py --config=configs/ae/{km_flow,rayleigh_benard,climate}.yaml
```
Latent model or baseline:
```
python train.py --config=configs/{km_flow,rayleigh_benard,climate}.yaml --model_name=interpolant --seed=42
```
Models: `interpolant`, `flow_matching`, `edm`, `ddpm`/`ddim`/`tsm`, `fno2d`, `sfno`. `ddpm`, `ddim` and `tsm`
share a training objective and differ only at sampling, so one checkpoint serves all three.

Useful options (also settable in the config):

| flag | meaning |
|:-|:-|
| `--optimizer` | `muon` (default; AdamW for non-matrix params) or `adam` |
| `--ema_decay` | EMA of the weights used for validation and checkpoints (default `0.999`) |
| `--max_steps`, `--val_every_n_steps`, `--save_every_n_train_steps`, `--archive_every_n_train_steps` | step-based schedule and checkpointing |
| `--dt_stride` | lead time in raw timesteps (Kolmogorov / Rayleigh-Benard) |

## Evaluation
Validation metrics on a checkpoint:
```
python val.py --config=configs/km_flow.yaml --model_name=interpolant --checkpoint=/path/to/model.ckpt --ensemble 8
```
NFE sweep with sampling wall-clock time (`--variants ode sde` for `edm`/`interpolant`):
```
python eval_nfe.py --config=configs/km_flow.yaml --model_name=interpolant \
    --checkpoint=/path/to/model.ckpt --nfes 2 5 10 20 --ensemble 8 --out=/path/to/results
```
Climate CRPS/SSR vs. lead time and 10-year climatological bias:
```
python eval_crps_leadtime.py --model_name=interpolant --variant=sde --sigma=0.5 --nfe=20 --checkpoint=/path/to/model.ckpt
python eval_bias_10yr.py --model_name=interpolant --variant=sde --sigma=0.5 --checkpoint=/path/to/model.ckpt
```
With an ensemble, deterministic metrics are computed on the ensemble mean, and `val/CRPS` (fair CRPS) and
`val/SSR` (spread/skill ratio) across members. Per-step metric curves are saved as `val_curves*.npz` in the run's log directory.

## Slurm
`launch.sh` builds the full experiment grid. It prints the `sbatch` commands and only submits with `--submit`:
```
./launch.sh train                                  # all datasets x models x seeds 42,43,44
./launch.sh eval --ckpt-root /path/to/data/logs    # NFE sweeps of the trained runs
./launch.sh crps                                   # climate CRPS/SSR vs lead time
./launch.sh bias                                   # climate 10-year bias
```
Pass cluster-specific `sbatch` flags through `SBATCH_ARGS`, e.g. `SBATCH_ARGS="-p gpu" ./launch.sh train --submit`.
