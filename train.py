# Default imports
import argparse
from datetime import datetime
import torch
import os 

# Custom imports
from common.utils import get_yaml, save_yaml
from lightning.pytorch.callbacks import LearningRateMonitor
from dataset.datamodule import PDEDataModule
from modules.train_module import TrainModule
from modules.ae_module import AutoencoderModule

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

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
    
    return config, modelconfig, trainconfig, dataconfig

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    
    pde = dataconfig['pde']
    description = args.description if args.description is not None else ""
    name = modelconfig["model_name"] + "_" + pde + "_" + description + "_" + str(seed) + "_" + now
    wandb_logger = WandbLogger(project=trainconfig["project"],
                               name=name,
                               mode=trainconfig["wandb_mode"])
    path = trainconfig["log_dir"] + name + "/"
    config['training']["log_dir"] = path

    os.makedirs(path, exist_ok=True) 
    save_yaml(config, path + "config.yml")

    if pde == "climate":
        monitor = "val/z500_240"
    else:
        monitor = "val/VRMSE"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        filename= "model_{epoch:02d}_best",
        mode='min',
        dirpath=path,
        save_last=True,
        save_top_k=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
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
                        callbacks=[checkpoint_callback, lr_monitor],
                        logger=wandb_logger,
                        num_sanity_val_steps=1,
                        accumulate_grad_batches=trainconfig.get("accumulate_grad_batches", 1),)
    
    if trainconfig["checkpoint"] is not None:
        trainer.fit(model=model,
                datamodule=datamodule,
                ckpt_path=trainconfig["checkpoint"])
    else:
        trainer.fit(model=model, 
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
    args = parser.parse_args()

    main(args)