"""Long climate rollout from a checkpoint, scored against reference climatological biases."""
import argparse
import os 
import torch 
import pickle 
from tqdm import tqdm
import numpy as np

from common.utils import get_yaml
from common.climate_utils import plot_result_climate, plot_loss, plot_spectrum, plot_climatological_bias
from common.loss import latitude_weighted_rmse
from dataset.datamodule import PDEDataModule
from dataset.plasim import SURFACE_FEATURES, MULTI_LEVEL_FEATURES
from modules.train_module import TrainModule

from lightning.pytorch import seed_everything


def main(args, model_path, save_path, bias_path, device='cuda', time_horizon=14612, plot_interval=4000):
    torch.set_float32_matmul_precision('high')
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = config["training"]["seed"]
    seed_everything(seed)

    checkpoint_path = model_path
    log_dir = save_path
    config['data']['batch_size'] = 1
    config["data"]["dataset"]["training_nsteps"] = 1
    config["data"]["dataset"]["val_nsteps"] = 1 
    ensemble_size = 1
    verbose = False

    os.makedirs(log_dir, exist_ok=True) 

    datamodule = PDEDataModule(config["data"])

    model = TrainModule(config=config,
                        normalizer=datamodule.normalizer)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    loader = datamodule.train_dataloader(shuffle=False)  # val split is too short for long rollouts
    surface_var_names = SURFACE_FEATURES 
    multilevel_var_names = MULTI_LEVEL_FEATURES
    
    t2m_losses = []
    pr_6h_losses = []
    z500_losses = []
    u250_losses = []
    t850_losses = []

    mean_pred_dict = {}
    mean_target_dict = {}

    model.to(device)
    z_pred = None
    i = 0
    loader.__len__ = lambda: time_horizon

    with torch.no_grad():
        if os.path.exists(log_dir + "/clim_predictions_final.pkl"):
            clim_pred_dict = pickle.load(open(log_dir + "/clim_predictions_final.pkl", "rb"))
            clim_target_dict = pickle.load(open(log_dir + "/clim_targets_final.pkl", "rb"))
            pred = clim_pred_dict
            print("Loading cached climatology predictions and targets.")
        else:
            for batch in tqdm(loader):
                batch = [item.to(device) for item in batch]
                if i == 0:
                    pass  # true initial condition
                else:  # feed back the previous prediction; forcings stay true
                    surface_feat, multi_level_feat, constants, yearly_constants, day_of_year, hour_of_day = batch 

                    surface_feat, multi_level_feat = model.normalizer.batch_denormalize(surface_feat, multi_level_feat)

                    for c, surface_feat_name in enumerate(surface_var_names):
                        surface_feat[:, 0, ... , c] = pred_feat_dict[surface_feat_name][:, -1]
                    for c, multi_level_feat_name in enumerate(multilevel_var_names):
                        multi_level_feat[:, 0, ..., c] = pred_feat_dict[multi_level_feat_name][:, -1]
                    
                    surface_feat, multi_level_feat = model.normalizer.batch_normalize(surface_feat, multi_level_feat)

                    batch = [surface_feat, multi_level_feat, constants, yearly_constants, day_of_year, hour_of_day]

                loss_dict, pred_feat_dict, target_feat_dict, z_pred = model.validation_step(batch, batch_idx=i, eval=True, z_pred=z_pred, ensemble_size=ensemble_size)
                
                if i == 0:
                    clim_pred_dict = {k: torch.zeros_like(v) for k, v in pred_feat_dict.items()}
                    clim_target_dict = {k: torch.zeros_like(v) for k, v in target_feat_dict.items()}

                    mean_pred_dict = {k: [] for k, v in pred_feat_dict.items()}
                    mean_target_dict = {k: [] for k, v in target_feat_dict.items()}

                for k in clim_pred_dict.keys():
                    clim_pred_dict[k] = clim_pred_dict[k] + pred_feat_dict[k]
                    clim_target_dict[k] = clim_target_dict[k] + target_feat_dict[k]

                for k in clim_pred_dict.keys():  # global mean
                    mean_pred_dict[k].append(torch.mean(pred_feat_dict[k]).item())
                    mean_target_dict[k].append(torch.mean(target_feat_dict[k]).item())

                t2m_loss = loss_dict['tas'].mean(0)
                pr_6h_loss = loss_dict['pr_6h'].mean(0)
                z500_loss = loss_dict['zg'][..., 7].mean(0)
                u250_loss = loss_dict['ua'][..., 4].mean(0)
                t850_loss = loss_dict['ta'][..., 10].mean(0)

                t2m_losses.append(t2m_loss[0].item())
                pr_6h_losses.append(pr_6h_loss[0].item())
                z500_losses.append(z500_loss[0].item())
                u250_losses.append(u250_loss[0].item())
                t850_losses.append(t850_loss[0].item())

                if (i+1) % plot_interval == 0 or i == 0 or i == time_horizon - 1:
                    pickle.dump(pred_feat_dict, open(log_dir + "/predictions_" + str(i) + ".pkl", "wb"))
                    pickle.dump(target_feat_dict, open(log_dir + "/targets_" + str(i) + ".pkl", "wb"))

                    pickle.dump(mean_pred_dict, open(log_dir + "/mean_predictions_" + str(i) + ".pkl", "wb"))
                    pickle.dump(mean_target_dict, open(log_dir + "/mean_targets_" + str(i) + ".pkl", "wb"))

                    pickle.dump(clim_pred_dict, open(log_dir + "/clim_predictions_" + str(i) + ".pkl", "wb"))
                    pickle.dump(clim_target_dict, open(log_dir + "/clim_targets_" + str(i) + ".pkl", "wb"))

                    if verbose:
                        print("Step: ", i+1)
                        print("T2M Loss: ", t2m_losses[-1])
                        print("PR_6H Loss: ", pr_6h_losses[-1])
                        print("Z500 Loss: ", z500_losses[-1])
                        print("U250 Loss: ", u250_losses[-1])
                        print("T850 Loss: ", t850_losses[-1])
                        print("--------------------------------------------------")

                    t2m_pred = pred_feat_dict['tas'][0].cpu().numpy()
                    t2m_target = target_feat_dict['tas'][0].cpu().numpy()
                    z500_pred = pred_feat_dict['zg'][0, ..., 7].cpu().numpy()
                    z500_target = target_feat_dict['zg'][0, ..., 7].cpu().numpy()
                    pr_6h_pred = pred_feat_dict['pr_6h'][0].cpu().numpy()
                    pr_6h_target = target_feat_dict['pr_6h'][0].cpu().numpy()
                    u250_pred = pred_feat_dict['ua'][0, ..., 4].cpu().numpy()
                    u250_target = target_feat_dict['ua'][0, ..., 4].cpu().numpy()
                    t850_pred = pred_feat_dict['ta'][0, ..., 10].cpu().numpy()
                    t850_target = target_feat_dict['ta'][0, ..., 10].cpu().numpy()

                    plot_result_climate(t2m_pred, # t h w
                                    t2m_target,
                                    f'{log_dir}/val_t2m_{i}.png',
                                    num_t=1)
                    plot_result_climate(z500_pred,
                                    z500_target,
                                    f'{log_dir}/val_z500_{i}.png',
                                    num_t=1)
                    plot_result_climate(pr_6h_pred,
                                    pr_6h_target,
                                    f'{log_dir}/val_pr_6h_{i}.png',
                                    num_t=1)
                    plot_result_climate(u250_pred,
                                    u250_target,
                                    f'{log_dir}/val_u250_{i}.png',
                                    num_t=1)
                    plot_result_climate(t850_pred,
                                    t850_target,
                                    f'{log_dir}/val_t850_{i}.png',
                                    num_t=1)
                    
                    plot_spectrum(t2m_pred,
                                    t2m_target,
                                    f'{log_dir}/val_t2m_spectrum_{i}.png',
                                    num_t=1)
                    plot_spectrum(z500_pred,
                                    z500_target,
                                    f'{log_dir}/val_z500_spectrum_{i}.png',
                                    num_t=1)
                    plot_spectrum(pr_6h_pred,
                                    pr_6h_target,
                                    f'{log_dir}/val_pr_6h_spectrum_{i}.png',
                                    num_t=1)
                    plot_spectrum(u250_pred,
                                    u250_target,
                                    f'{log_dir}/val_u250_spectrum_{i}.png',
                                    num_t=1)
                    plot_spectrum(t850_pred,
                                    t850_target,
                                    f'{log_dir}/val_t850_spectrum_{i}.png',
                                    num_t=1)
                    
                    plot_loss(t2m_losses,
                            f'{log_dir}/t2m_loss.png',
                            key='T2M')
                    plot_loss(pr_6h_losses,
                            f'{log_dir}/pr_6h_loss.png',
                                key='PR_6H')
                    plot_loss(z500_losses,
                                f'{log_dir}/z500_loss.png',
                                    key='Z500')
                    plot_loss(u250_losses,
                                f'{log_dir}/u250_loss.png',
                                key='U250')
                    plot_loss(t850_losses,
                                f'{log_dir}/t850_loss.png',
                                key='T850')
                    
                    pickle.dump(t2m_losses, open(log_dir + "/t2m_loss.pkl", "wb"))
                    pickle.dump(pr_6h_losses, open(log_dir + "/pr_6h_loss.pkl", "wb"))
                    pickle.dump(z500_losses, open(log_dir + "/z500_loss.pkl", "wb"))
                    pickle.dump(u250_losses, open(log_dir + "/u250_loss.pkl", "wb"))
                    pickle.dump(t850_losses, open(log_dir + "/t850_loss.pkl", "wb"))

                i = i + 1
                if i == time_horizon:
                    break
            
            clim_pred_dict = {k: v/time_horizon for k, v in clim_pred_dict.items()}
            clim_target_dict = {k: v/time_horizon for k, v in clim_target_dict.items()}

            pred = clim_pred_dict

            pickle.dump(clim_pred_dict, open(log_dir + "/clim_predictions_final.pkl", "wb"))
            pickle.dump(clim_target_dict, open(log_dir + "/clim_targets_final.pkl", "wb"))

        file_key_list = ['tas', "pr_6h", "zg_50000.0", "ua_25000.0", "ta_85000.0", "hus_85000.0"]
        dict_key_list = ['tas', "pr_6h", "zg", "ua", "ta", "hus"]
        result_dict = {}
        for f_k, d_k in zip(file_key_list, dict_key_list):
            path = f"{bias_path}/{f_k}_bias.npy"
            bias = np.load(path)  # nlat nlon
            bias = torch.tensor(bias).unsqueeze(0).unsqueeze(0)  # b t nlat nlon
            
            l = -1 
            if f_k == "zg_50000.0":
                l = 7
            elif f_k == "ua_25000.0":
                l = 4
            elif f_k == "ta_85000.0" or f_k == "hus_85000.0":
                l = 10
            
            if l == -1:
                pred_k = pred[d_k].cpu()
            else:
                pred_k = pred[d_k][..., l].cpu()

            loss = latitude_weighted_rmse(pred_k, 
                                        bias,
                                        with_poles=False,
                                        nlon=128,
                                        nlat=64,)
            result_dict[f_k] = loss.item()
            plot_climatological_bias(pred_k[0, 0], bias[0, 0], save_path=f"{log_dir}/{d_k}_bias.png")
        
        print(result_dict)
        with open(log_dir + "/climatology_results.txt", "w") as f:
            for k, v in result_dict.items():
                f.write(f"{k}: {v}\n")

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
    parser = argparse.ArgumentParser(description='Climatological bias evaluation')
    parser.add_argument("--config", default=None)
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--devices', nargs='+', default=[])
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--model_dir', default=None, help='Directory containing model checkpoints')
    parser.add_argument('--wandb_mode', default=None)
    parser.add_argument('--description', default=None)
    parser.add_argument('--checkpoint', default=None, help='Path to the checkpoint to resume training')
    parser.add_argument('--num_refinement_steps', type=int, default=None, help='Number of refinement steps')
    parser.add_argument('--num_ddim_steps', type=int, default=None, help='Number of ddim steps')
    parser.add_argument('--num_edm_steps', type=int, default=None, help='Number of edm steps')
    parser.add_argument('--skip_percent', type=float, default=None, help='Skip percent of TSM')
    parser.add_argument('--mode', type=str, default='single')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--bias_path', type=str, default='/path/to/data/bias')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    bias_path = args.bias_path
    device = args.device
    time_horizon = 146095  # 6 h steps: 14612 = 10 yr, 146095 = 100 yr
    plot_interval = 40000

    if args.mode == 'single':
        path = args.model_path
        print(f"Evaluating model: {path}")
        save_path = path.replace(".ckpt", f"_climatology_euler_100yr")
        main(args, 
            path, 
            save_path,
            bias_path, 
            device=device, 
            time_horizon=time_horizon, 
            plot_interval=plot_interval)
    else:  # every checkpoint in --model_dir
        model_dir = args.model_dir
        model_paths = []
        device = "cuda"
        for file in os.listdir(model_dir):
            if file.endswith(".ckpt"):
                model_paths.append(os.path.join(model_dir, file))
        for path in model_paths:
            save_path = path.replace(".ckpt", f"_climatology_euler_100yr")
            main(args, 
                path, 
                save_path, 
                bias_path, 
                device=device, 
                time_horizon=time_horizon, 
                plot_interval=plot_interval)
