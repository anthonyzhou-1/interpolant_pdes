# Default imports
import argparse
from datetime import datetime
import itertools
import torch
from torch.optim.swa_utils import get_ema_avg_fn
import os 

# Custom imports
from common.utils import get_yaml, save_yaml
from lightning.pytorch.callbacks import LearningRateMonitor, WeightAveraging
from dataset.datamodule import PDEDataModule
from modules.train_module import TrainModule
from modules.ae_module import AutoencoderModule

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class EMAWeightAveraging(WeightAveraging):
    """EMA of the weights, updated every optimizer step.

    The EMA weights are used for validation and saved as the checkpoint's "state_dict";
    the raw weights are kept under "current_model_state". Buffers are constants, so only
    parameters are averaged.
    """

    def __init__(self, decay=0.999):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay), use_buffers=False)

    def should_update(self, step_idx=None, epoch_idx=None):
        return True

    # Pair live and averaged tensors by name (not position), skipping tensors registered
    # after the shadow copy was made.
    def _pairs(self, pl_module):
        average = dict(itertools.chain(self._average_model.module.named_parameters(),
                                       self._average_model.module.named_buffers()))
        for name, current in itertools.chain(pl_module.named_parameters(),
                                             pl_module.named_buffers()):
            avg = average.get(name)
            if avg is None or avg.shape != current.shape:
                continue
            yield avg, current

    def _swap_models(self, pl_module):
        for avg, current in self._pairs(pl_module):
            tmp = avg.data.clone()
            avg.data.copy_(current.data)
            current.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module):
        for avg, current in self._pairs(pl_module):
            current.data.copy_(avg.data)


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
    if args.dt_stride is not None:
        dataconfig['dataset']['dt_stride'] = args.dt_stride
    if args.max_steps is not None:
        trainconfig["max_steps"] = args.max_steps
    if args.save_every_n_train_steps is not None:
        trainconfig["save_every_n_train_steps"] = args.save_every_n_train_steps
    if args.archive_every_n_train_steps is not None:
        trainconfig["archive_every_n_train_steps"] = args.archive_every_n_train_steps
    if args.val_every_n_steps is not None:
        trainconfig["val_every_n_steps"] = args.val_every_n_steps
    if args.ema_decay is not None:
        trainconfig["ema_decay"] = args.ema_decay
    if args.optimizer is not None:
        modelconfig["optimizer"] = args.optimizer
    
    return config, modelconfig, trainconfig, dataconfig

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed, workers=True)
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

    callbacks = []

    # last.ckpt (for resuming) is refreshed every save_every_n_train_steps; a permanent
    # checkpoint is archived every archive_every_n_train_steps.
    save_every_n_train_steps = int(trainconfig.get("save_every_n_train_steps", 1000))
    archive_every_n_train_steps = trainconfig.get("archive_every_n_train_steps", 50000)

    last_checkpoint = ModelCheckpoint(
        dirpath=path,
        every_n_train_steps=save_every_n_train_steps,
        save_last=True,
        save_top_k=0,
    )
    callbacks.append(last_checkpoint)
    print(f"Refreshing last.ckpt every {save_every_n_train_steps} gradient steps")

    if archive_every_n_train_steps is not None and int(archive_every_n_train_steps) > 0:
        archive_checkpoint = ModelCheckpoint(
            dirpath=path,
            filename="model_step_{step:08d}",
            auto_insert_metric_name=False,
            every_n_train_steps=int(archive_every_n_train_steps),
            save_last=False,
            save_top_k=-1,
        )
        callbacks.append(archive_checkpoint)
        print(f"Archiving a permanent checkpoint every {int(archive_every_n_train_steps)} "
              f"gradient steps")

    if trainconfig.get("save_best", True):
        best_checkpoint = ModelCheckpoint(
            monitor=monitor,
            filename="model_step_{step:08d}_best",
            auto_insert_metric_name=False,
            mode='min',
            dirpath=path,
            save_last=False,
            save_top_k=1,
        )
        callbacks.append(best_checkpoint)
        print(f"Keeping the best checkpoint by {monitor}")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Set training.ema_decay to null (or 0) to disable EMA.
    ema_decay = trainconfig.get("ema_decay", None)
    if ema_decay is not None and float(ema_decay) > 0:
        callbacks.append(EMAWeightAveraging(decay=float(ema_decay)))
        print(f"Averaging weights with EMA decay {float(ema_decay)}")

    max_steps = trainconfig.get("max_steps", None)
    if max_steps is not None and int(max_steps) > 0:
        max_steps = int(max_steps)
        max_epochs = -1
        print(f"Training for {max_steps} gradient steps (max_epochs disabled)")
    else:
        max_steps = -1
        max_epochs = trainconfig.get("max_epochs", -1)
        print(f"Training for {max_epochs} epochs")

    accumulate_grad_batches = trainconfig.get("accumulate_grad_batches", 1)

    # val_check_interval counts batches, so scale by the accumulation factor;
    # check_val_every_n_epoch=None lets the counter run across epochs.
    val_every_n_steps = trainconfig.get("val_every_n_steps", None)
    if val_every_n_steps is not None and int(val_every_n_steps) > 0:
        val_every_n_steps = int(val_every_n_steps)
        val_kwargs = {"check_val_every_n_epoch": None,
                      "val_check_interval": val_every_n_steps * accumulate_grad_batches}
        print(f"Validating every {val_every_n_steps} gradient steps")
    else:
        every_n_epoch = trainconfig.get("check_val_every_n_epoch", 1)
        val_kwargs = {"check_val_every_n_epoch": every_n_epoch}
        print(f"Validating every {every_n_epoch} epoch(s)")

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
                        log_every_n_steps = trainconfig["log_every_n_steps"],
                        max_epochs = max_epochs,
                        max_steps = max_steps,
                        default_root_dir = path,
                        callbacks=callbacks,
                        logger=wandb_logger,
                        num_sanity_val_steps=trainconfig.get("num_sanity_val_steps", 1),
                        accumulate_grad_batches=accumulate_grad_batches,
                        **val_kwargs,)
    
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
    parser.add_argument('--dt_stride', type=int, default=None, help='Prediction lead time in raw timesteps (rayleigh_benard, km_flow)')
    parser.add_argument('--max_steps', type=int, default=None, help='Stop after N gradient steps (overrides max_epochs).')
    parser.add_argument('--save_every_n_train_steps', type=int, default=None, help='Refresh last.ckpt every N gradient steps.')
    parser.add_argument('--archive_every_n_train_steps', type=int, default=None, help='Keep a permanent checkpoint every N gradient steps (0 disables).')
    parser.add_argument('--val_every_n_steps', type=int, default=None, help='Validate every N gradient steps.')
    parser.add_argument('--ema_decay', type=float, default=None, help='EMA decay for the averaged weights (0 disables EMA).')
    parser.add_argument('--optimizer', default=None, choices=['muon', 'adam'], help='Optimizer (default: model.optimizer).')
    args = parser.parse_args()

    main(args)
