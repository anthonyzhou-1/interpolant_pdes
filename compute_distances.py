import torch 
from einops import rearrange
import argparse
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from lightning.pytorch import seed_everything

from common.utils import get_yaml
from dataset.dataset_2D import PDEDataset2D
from dataset.normalizer import ScalarNormalizer
from common.distances import johnson_lindenstrauss, sliced_wasserstein_distance,  \
    compute_rbf_mmd_median_heuristic, c2st_nn, color_dict, generate_palette, plot_distances

def get_dataset_rb(dataconfig):
    dataset_config = dataconfig["dataset"]
    normalizer_config = dataconfig["normalizer"]

    base_path = dataset_config["base_path"]
    pde = "rayleigh_benard"
    stat_path = normalizer_config["stat_path"]

    dataset = WellDataset(
        well_base_path=f"{base_path}/datasets",
        well_dataset_name=pde,
        well_split_name="train",
        n_steps_input=1,
        n_steps_output=1,
        use_normalization=True,
        normalization_type = ZScoreNormalization,
        normalization_path=stat_path,
        full_trajectory_mode = True,
        min_dt_stride=2, 
        max_dt_stride=2,
        include_filters=["Rayleigh_1e8_Prandtl_10"]
    )

    return dataset

def get_dataset_km_flow(dataconfig):
    dataset_config = dataconfig["dataset"]
    normalizer_config = dataconfig["normalizer"]
    stat_path = normalizer_config["stat_path"]
    train_path = dataset_config["train_path"]
    resolution = [100, 160, 160]

    normalizer = ScalarNormalizer(stat_path=stat_path)

    dataset = PDEDataset2D(path = train_path,
                            split = "train",
                            resolution = resolution,
                            normalizer = normalizer,
                            return_traj=True)
    
    return dataset
    
def process_args(args, config):
    modelconfig = config['model']
    trainconfig = config['training']
    dataconfig = config['data']

    if len(args.devices) > 0:
        trainconfig["devices"] = [int(device) for device in args.devices]
    if args.wandb_mode is not None:
        trainconfig["wandb_mode"] = args.wandb_mode
    if args.model_name is not None:
        modelconfig["model_name"] = args.model_name
    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint
    
    return config, modelconfig, trainconfig, dataconfig

def get_sequential_distances(dist, noise, seed=1):
    """
    Computes distances between a sequential distribution and noise.    
    Args:
        dist (torch.Tensor): Sequential distribution of shape (num_dists, n_samples, n_features).
        noise (torch.Tensor): Noise distribution of shape (n_samples, n_features).
        seed (int): Random seed for reproducibility.
    """

    sequential_distances_sw = []
    gaussian_distance_sw = []
    for i in range(dist.shape[0] - 1):
        distance_sw = sliced_wasserstein_distance(dist[i], dist[i+1], num_projections=300)
        sequential_distances_sw.append(distance_sw.item())
        noise_distance_sw = sliced_wasserstein_distance(dist[i+1], noise, num_projections=300)
        gaussian_distance_sw.append(noise_distance_sw.item())

    sequential_distances_c2st = []
    gaussian_distance_c2st = []
    for i in range(dist.shape[0] - 1):
        distance_c2st = c2st_nn(dist[i], dist[i+1], seed=seed)
        sequential_distances_c2st.append(distance_c2st.item())
        noise_distance_c2st = c2st_nn(dist[i+1], noise, seed=seed)
        gaussian_distance_c2st.append(noise_distance_c2st.item())
    
    sequential_distances_mmd = []
    gaussian_distance_mmd = []
    for i in range(dist.shape[0] - 1):
        distance_mmd = compute_rbf_mmd_median_heuristic(dist[i], dist[i+1])
        sequential_distances_mmd.append(distance_mmd.item())
        noise_distance_mmd = compute_rbf_mmd_median_heuristic(dist[i+1], dist)
        gaussian_distance_mmd.append(noise_distance_mmd.item())
    
    return_dict = {
        "sequential_distances_sw": sequential_distances_sw,
        "gaussian_distance_sw": gaussian_distance_sw,
        "sequential_distances_c2st": sequential_distances_c2st,
        "gaussian_distance_c2st": gaussian_distance_c2st,
        "sequential_distances_mmd": sequential_distances_mmd,
        "gaussian_distance_mmd": gaussian_distance_mmd
    }

    return return_dict

def plot_all(data, pde, save_path):
    for i in range(5):
        key = f"trial_{i}"
        data_dict = data[key]
        sequential_sw_distances.append(torch.tensor(data_dict['sequential_distances_sw']))
        gaussian_sw_distances.append(torch.tensor(data_dict['gaussian_distance_sw']))
        sequential_mmd_distances.append(torch.tensor(data_dict['sequential_distances_mmd']))
        gaussian_mmd_distances.append(torch.tensor(data_dict['gaussian_distance_mmd']))
        sequential_c2st_distances.append(torch.tensor(data_dict['sequential_distances_c2st']))
        gaussian_c2st_distances.append(torch.tensor(data_dict['gaussian_distance_c2st']))
        
    sequential_sw_distances = torch.stack(sequential_sw_distances)
    sequential_mmd_distances = torch.stack(sequential_mmd_distances)
    sequential_c2st_distances = torch.stack(sequential_c2st_distances)
    gaussian_sw_distances = torch.stack(gaussian_sw_distances)
    gaussian_mmd_distances = torch.stack(gaussian_mmd_distances)
    gaussian_c2st_distances = torch.stack(gaussian_c2st_distances)
    num_dists = torch.arange(len(sequential_sw_distances[0]))

    col_dark = {}
    col_light = {}
    experiments = ["wasserstein", "mmd", "c2st"]
    for e, exp_name in enumerate(experiments):
        col_dark[exp_name] = generate_palette(color_dict[exp_name], saturation='dark')[2]
        col_light[exp_name] = generate_palette(color_dict[exp_name], saturation='light')[-1]
        
    color_list = [col_light, col_dark]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex='col')
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    for ax in axes.flatten():
        # move spines outward
        ax.spines['bottom'].set_position(('outward', 4))
        ax.spines['left'].set_position(('outward', 4))
        ax.locator_params(nbins=4)

    for i in range(3):
        if i == 0:
            plot_distances(num_dists,
                        sequential_sw_distances.mean(dim=0),
                        2*sequential_sw_distances.std(dim=0),
                        metric_name="sw",
                        dataset_name=pde,
                        ax=axes[i],
                        color=color_list[0]['wasserstein'],
                        label="D(t, t+1)",)
            plot_distances(num_dists,
                        gaussian_sw_distances.mean(dim=0),
                        2*gaussian_sw_distances.std(dim=0),
                        metric_name="sw",
                        dataset_name=pde,
                        ax=axes[i],
                        color=color_list[1]['wasserstein'],
                        label="D(t+1, N)")
        elif i == 1:
            plot_distances(num_dists,
                        sequential_c2st_distances.mean(dim=0),
                        2*sequential_c2st_distances.std(dim=0),
                        metric_name="c2st",
                        dataset_name=pde,
                        ax=axes[i],
                        color=color_list[0]['c2st'],
                        label="D(t, t+1)")
            plot_distances(num_dists,
                        gaussian_c2st_distances.mean(dim=0),
                        2*gaussian_c2st_distances.std(dim=0),
                        metric_name="c2st",
                        dataset_name=pde,
                        ax=axes[i],
                        color=color_list[1]['c2st'],
                        label="D(t+1, N)")
        else:
            plot_distances(num_dists,
                        sequential_mmd_distances.mean(dim=0),
                        2*sequential_mmd_distances.std(dim=0),
                        metric_name="mmd",
                        dataset_name=pde,
                        ax=axes[i],
                        color=color_list[0]['mmd'],
                        label="D(t, t+1)")
            plot_distances(num_dists,
                        gaussian_mmd_distances.mean(dim=0),
                        2*gaussian_mmd_distances.std(dim=0),
                        metric_name="mmd",
                        dataset_name=pde,
                        ax=axes[i],
                        color=color_list[1]['mmd'],
                        label="D(t+1, N)")
            axes[i].set_yscale('log')
            
        ymin_current, ymax_current = axes[i].get_ylim()
        axes[i].set_ylim([ymin_current, 1.1 * ymax_current])
        if i == 2:
            axes[i].set_ylim([ymin_current, 5 * ymax_current])
        axes[i].legend(loc='upper right', fontsize=12, frameon=False)

    axes[0].set_title("Sliced Wasserstein Distance", fontsize=18)
    axes[1].set_title("Classifier 2-Sample Test", fontsize=18)
    axes[2].set_title("Maximum Mean Discrepancy", fontsize=18)

    fig.suptitle(f"{pde} Dataset", fontsize=20, y=1.03)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)
    log_dir = trainconfig["log_dir"]
    seed = trainconfig["seed"]
    seed_everything(seed)

    pde = dataconfig['pde']

    if pde == "rayleigh_benard":
        dataset = get_dataset_rb(dataconfig)
        n_samples = 40
    elif pde == "km_flow":
        dataset = get_dataset_km_flow(dataconfig)
        n_samples = 1000

    k = 300
    num_trials = 5
    fold_pct = 0.8
    samples_per_trial = int(n_samples * fold_pct)

    all_data = []   
    for i in range(n_samples):
        batch = dataset.__getitem__(i)
        data = torch.tensor(batch['output_fields'][..., 0]) # (n_t, n_x, n_y)
        data_flat = data.view(data.shape[0], -1) # (n_t, D)
        all_data.append(data_flat)
    all_data = torch.stack(all_data) # (num_samples, n_t, D)
    all_data = rearrange(all_data, 'n t d-> t n d') # (n_t, num_samples, D)

    noise = torch.rand((n_samples, all_data.shape[-1]))
    all_data_flat = rearrange(all_data, 't n d -> (t n) d')  # (n_t * num_samples, D)
    data_noised = torch.cat([noise, all_data_flat], dim=0)  # (n_samples + n_t * num_samples, D)

    data_JL_noised = johnson_lindenstrauss(data_noised, k) # (n_samples + n_t * num_samples, k)
    noise_JL = data_JL_noised[:n_samples] # (n_samples, k)
    data_JL = data_JL_noised[n_samples:] # (n_t * num_samples, k)
    data_JL = rearrange(data_JL, '(t n) k -> t n k', n=n_samples)  # (n_t, num_samples, k)
    
    trial_dict = {}

    for i in tqdm(range(num_trials)):
        perm = torch.randperm(n_samples)
        idx = perm[:samples_per_trial]
        sequential_distances_dict = get_sequential_distances(data_JL[:, idx], noise_JL[idx], seed=i)
        trial_dict[f'trial_{i}'] = sequential_distances_dict

    output_path = f"{log_dir}sequential_distances_{pde}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(trial_dict, f)
    
    plot_all(trial_dict, pde, f"{log_dir}sequential_distances_{pde}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--config", default=None)
    parser.add_argument('--devices', nargs='+', help='<Required> Set flag', default=[])
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--wandb_mode', default=None)
    parser.add_argument('--description', default=None)
    parser.add_argument('--checkpoint', default=None, help='Path to the checkpoint to resume training')
    args = parser.parse_args()

    main(args)