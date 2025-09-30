# Default imports
import argparse
from datetime import datetime
import torch
import os 

# Custom imports
from common.utils import get_yaml, save_yaml
from dataset.datamodule import PDEDataModule
from modules.train_module import TrainModule
from modules.ae_module import AutoencoderModule

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

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

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    pde = dataconfig['pde']

    description = args.description if args.description is     not None else ""
    name = modelconfig["model_name"] + "_" + pde + "_" + description + "_" + str(seed) + "_" + now + "VAL"
    wandb_logger = WandbLogger(project=trainconfig["project"],
                               name=name,
                               mode=trainconfig["wandb_mode"])
    path = trainconfig["log_dir"] + name + "/"
    config['training']["log_dir"] = path

    os.makedirs(path, exist_ok=True) 
    save_yaml(config, path + "config.yml")
    
    datamodule = PDEDataModule(dataconfig=dataconfig)
    
    if modelconfig['model_name'] == "AE":
        model = AutoencoderModule(config,
                                  normalizer=datamodule.normalizer)
    else:
        model = TrainModule(config,
                            normalizer=datamodule.normalizer)

    trainer = L.Trainer(devices = trainconfig["devices"],
                        accelerator = trainconfig["accelerator"],
                        strategy = trainconfig["strategy"],
                        check_val_every_n_epoch = trainconfig["check_val_every_n_epoch"],
                        log_every_n_steps = trainconfig["log_every_n_steps"],
                        max_epochs = trainconfig["max_epochs"],
                        default_root_dir = path,
                        logger=wandb_logger,
                        num_sanity_val_steps=0)
    
    if trainconfig["checkpoint"] is not None:
        trainer.validate(model=model,
                datamodule=datamodule,
                ckpt_path=trainconfig["checkpoint"])
    else:
        trainer.validate(model=model, 
                datamodule=datamodule)

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

    args = parser.parse_args()

    main(args)