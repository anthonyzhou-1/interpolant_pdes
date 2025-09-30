# Default imports
import argparse
import os 
import torch 
from tqdm import tqdm
import pickle 
import numpy as np

# Custom imports
from common.utils import get_yaml
from common.climate_utils import plot_result_climate, plot_crps, plot_ssr
from dataset.datamodule import PDEDataModule
from modules.train_module import TrainModule

from lightning.pytorch import seed_everything
import xarray as xr 
import weatherbench2.metrics as wb_metrics

ensemble_mean = wb_metrics.EnsembleMeanMSE()
ensemble_variance = wb_metrics.EnsembleVariance()
crps = wb_metrics.CRPS()

def get_latitude(longitude_resolution=128, latitude_resolution=64):
    lat_end = (latitude_resolution-1)*(360/longitude_resolution) / 2
    latitude = np.linspace(-lat_end, lat_end, latitude_resolution)
    return latitude

def get_ssr_chunk(pred, target, latitude):

    forecast = xr.Dataset(
        {
            "var": (["realization", "time", "latitude", "longitude"], pred),
        },
        coords={"latitude": latitude,}
    )

    truth = xr.Dataset(
        {
            "var": (["time", "latitude", "longitude"], target),
        },
        coords={"latitude": latitude,}
    )

    mse = ensemble_mean.compute_chunk(forecast, truth)
    variance = ensemble_variance.compute_chunk(forecast, truth)

    mse_value = mse["var"].values
    variance_value = variance["var"].values

    ssr = np.sqrt(variance_value)/np.sqrt(mse_value)

    return ssr 

def get_crps_chunk(pred, target, latitude):

    forecast = xr.Dataset(
        {
            "var": (["realization", "time", "latitude", "longitude"], pred),
        },
        coords={"latitude": latitude,}
    )

    truth = xr.Dataset(
        {
            "var": (["time", "latitude", "longitude"], target),
        },
        coords={"latitude": latitude,}
    )
    out = crps.compute_chunk(forecast, truth)
    return out["var"]

def main(args, model_path, save_path, device='cuda'):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)
    torch.set_float32_matmul_precision('high') # to use tensor cores if available
    seed = config["training"]["seed"]
    seed_everything(seed)

    checkpoint_path = model_path
    log_dir = save_path
    config['data']['batch_size'] = 1
    time_horizon = 120 #40 for 10day, 120 for 30day, 240 for 60day
    config["data"]["dataset"]["val_nsteps"] = time_horizon
    ensemble_size = 32
    plot_interval = 1
    sample_interval = 12 # evaluate every 3 days, which is 4*3=12 steps
    save_out=False
    num_t = 6

    latitude = get_latitude(longitude_resolution=128, latitude_resolution=64)
    idx_dict = {
        "t2m": 6,
        "pr_6h": 5,
        "z500": 12 + 7*5,
        "u200": 10 + 3*5,
        "hus850": 8 + 10*5
    }

    os.makedirs(log_dir, exist_ok=True) 

    datamodule = PDEDataModule(config["data"])

    model = TrainModule(config=config,
                        normalizer=datamodule.normalizer)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    loader = datamodule.val_dataloader()
    model.to(device)

    crps_dict = {"t2m": [], "pr_6h": [], "z500": [], "u200": [], "hus850": []}
    ssr_dict = {"t2m": [], "pr_6h": [], "z500": [], "u200": [], "hus850": []}

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader)):
            if (idx+1) % sample_interval != 0 and idx != 0:
                print(f"Skipping batch {idx} for evaluation")
                continue 
            
            if os.path.exists(os.path.join(log_dir, f"crps_t2m_{idx}.pkl")):
                print(f"Loading cached result {idx}")
                for key, channel in idx_dict.items():
                    with open(os.path.join(log_dir, f"crps_{key}_{idx}.pkl"), "rb") as f:
                        crps_channel = pickle.load(f)
                    with open(os.path.join(log_dir, f"ssr_{key}_{idx}.pkl"), "rb") as f:
                        ssr_channel = pickle.load(f)
                    crps_dict[key].append(crps_channel)
                    ssr_dict[key].append(ssr_channel)
                
                continue

            batch = [item.to(device) for item in batch]
            
            _, pred, target, _ = model.validation_step(batch, batch_idx=idx, eval=True, ensemble_size=ensemble_size, return_ens=True)
            
            pred = pred.cpu()
            target = target.cpu()
            # pred in shape b ens t nlat nlon (c+nlevel*c)
            # target in shape b t nlat nlon (c+nlevel*c)

            if save_out:
                with open(os.path.join(log_dir, f"pred_{idx}.pkl"), "wb") as f:
                    pickle.dump(pred.cpu(), f)
                with open(os.path.join(log_dir, f"target_{idx}.pkl"), "wb") as f:
                    pickle.dump(target.cpu(), f)

            for key, channel in idx_dict.items():
                # shape of crps, ssr is (t,)
                crps_channel = get_crps_chunk(pred[0, ..., channel], target[0, ..., channel], latitude=latitude)
                ssr_channel = get_ssr_chunk(pred[0, ..., channel], target[0, ..., channel], latitude=latitude)
                crps_dict[key].append(crps_channel)
                ssr_dict[key].append(ssr_channel)

                with open(os.path.join(log_dir, f"crps_{key}_{idx}.pkl"), "wb") as f:
                    pickle.dump(crps_channel, f)
                with open(os.path.join(log_dir, f"ssr_{key}_{idx}.pkl"), "wb") as f:
                    pickle.dump(ssr_channel, f)

                if (idx+1) % plot_interval == 0:
                    plot_result_climate(pred[0, 0, ..., channel].cpu().numpy(),
                                        target[0, ..., channel].cpu().numpy(),
                                        os.path.join(log_dir, f"{key}_{idx}.png"),
                                        num_t=num_t,
                                        cmap='twilight_shifted')
                    
                    plot_crps(crps_channel,
                            title=key,
                            save_path= os.path.join(log_dir, f"crps_{key}_{idx}.png"))
                    plot_ssr(ssr_channel,
                            title=key,
                            save_path= os.path.join(log_dir, f"ssr_{key}_{idx}.png"))
        
    # save the crps_dict and ssr_dict
    for key in crps_dict.keys():
        crps_dict[key] = np.array(crps_dict[key])
        ssr_dict[key] = np.array(ssr_dict[key])

    with open(os.path.join(log_dir, "crps_dict.pkl"), "wb") as f:
        pickle.dump(crps_dict, f)
    with open(os.path.join(log_dir, "ssr_dict.pkl"), "wb") as f:
        pickle.dump(ssr_dict, f)
    
    # plot the crps_dict and ssr_dict
    for key in crps_dict.keys():
        time_averaged_crps = crps_dict[key].mean(axis=0)
        time_averaged_ssr = ssr_dict[key].mean(axis=0)
        plot_crps(time_averaged_crps,
                  title=key,
                  save_path=os.path.join(log_dir, f"crps_{key}_time_averaged.png"))
        plot_ssr(time_averaged_ssr,
                    title=key,
                    save_path=os.path.join(log_dir, f"ssr_{key}_time_averaged.png"))

def process_args(args, config):
    modelconfig = config['model']
    trainconfig = config['training']
    dataconfig = config['data']

    if len(args.devices) > 0:
        trainconfig["devices"] = [int(device) for device in args.devices]
    if args.seed is not None:
        trainconfig["seed"] = args.seed
    if args.wandb_mode is not None:
        trainconfig["wandb_mode"] = args.wandb_mode
    if args.model_name is not None:
        modelconfig["model_name"] = args.model_name
    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint
    if args.num_refinement_steps is not None:
        modelconfig['flow_matching']['num_refinement_steps'] = args.num_refinement_steps
        modelconfig['interpolant']['num_refinement_steps'] = args.num_refinement_steps
    if args.num_ddim_steps is not None:
        modelconfig['ddim']['num_ddim_steps'] = args.num_ddim_steps
    if args.num_edm_steps is not None:
        modelconfig['edm']['num_steps'] = args.num_edm_steps
    if args.skip_percent is not None:
        modelconfig['tsm']['skip_percent'] = args.skip_percent
    
    return config, modelconfig, trainconfig, dataconfig
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--config", default=None)
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--devices', nargs='+', help='<Required> Set flag', default=[])
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--wandb_mode', default=None)
    parser.add_argument('--description', default=None)
    parser.add_argument('--checkpoint', default=None, help='Path to the checkpoint to resume training')
    parser.add_argument('--num_refinement_steps', type=int, default=None, help='Number of refinement steps')
    parser.add_argument('--num_ddim_steps', type=int, default=None, help='Number of ddim steps')
    parser.add_argument('--num_edm_steps', type=int, default=None, help='Number of edm steps')
    parser.add_argument('--skip_percent', type=float, default=None, help='Skip percent of TSM')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    args = parser.parse_args()
    
    #interpolant
    #model_path = "logs/interpolant_climate__2025-07-30T16-40-19/epoch=51-step=158288.ckpt"
    #save_path = "logs/interpolant_climate__2025-07-30T16-40-19/epoch=51-step=158288_CRPS"

    #flow matching
    #model_path = "/home/ayz2/climate_diffusion/logs/ClimaDiT_ldm_base_32_ddp_2025-06-19T16-56-18/model_epoch=47_fixed.ckpt"
    #save_path = "/home/ayz2/climate_diffusion/logs/ClimaDiT_ldm_base_32_ddp_2025-06-19T16-56-18/CRPS_NEW"

    #edm
    #model_path = "logs/edm_climate__42_2025-08-31T19-27-37/epoch=39-step=121760.ckpt"
    #save_path = "logs/edm_climate__42_2025-08-31T19-27-37/epoch=39-step=121760_CRPS"
    
    model_path = args.model_path
    save_path = args.save_path
    device = args.device
    main(args, model_path, save_path, device=device)