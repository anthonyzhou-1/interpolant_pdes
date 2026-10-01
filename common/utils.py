import yaml
from inspect import isfunction
import torch 
from inspect import signature
from typing import Any, Callable
import torch.nn as nn
import os
import torch.distributed

def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))      

def get_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_yaml(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def build_kwargs_from_config(config: dict, target_func: Callable):
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def list_join(x: list, sep="\t", format_str="%s") -> str:
    return sep.join([format_str % val for val in x])


def list_sum(x: list) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])

def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def list_mean(x: list) -> Any:
    return list_sum(x) / len(x)

def get_device(model: nn.Module) -> torch.device:
    return model.parameters().__next__().device

def is_dist_initialized() -> bool:
    return torch.distributed.is_initialized()

def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])

def get_dist_rank() -> int:
    return int(os.environ["RANK"])

def is_master() -> bool:
    return get_dist_rank() == 0

def sync_tensor(tensor: torch.Tensor | float, reduce="mean"):
    if not is_dist_initialized():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list
    

    
    


# --- Muon ---------------------------------------------------------------------
# Attributes holding the transformer body (DiT / ClimaDiT). Models without one use Adam.
MUON_BODY_MODULE_NAMES = ("sa_blocks", "fa_blocks", "ca_blocks")

# Zero-initialized adaLN modulators stay on AdamW.
NO_MUON_PARAM_KEYS = ("adaLN", "modulation", "modulator")


def muon_param_groups(model, lr, muon_lr_mult=10.0, weight_decay=0.01, betas=(0.9, 0.95)):
    """Split `model` into a Muon group (hidden real 2D+ body matrices) and an AdamW group
    (everything else). Returns None if the model has no transformer body.
    """
    body_modules = [getattr(model, name) for name in MUON_BODY_MODULE_NAMES
                    if hasattr(model, name)]
    if not body_modules:
        return None

    hidden_weights, hidden_ids = [], set()
    for mod in body_modules:
        for name, p in mod.named_parameters():
            if (p.requires_grad and p.ndim >= 2 and not p.is_complex()
                    and id(p) not in hidden_ids
                    and not any(k in name for k in NO_MUON_PARAM_KEYS)):
                hidden_weights.append(p)
                hidden_ids.add(id(p))
    if not hidden_weights:
        return None

    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in hidden_ids]

    return [
        dict(params=hidden_weights, use_muon=True,
             lr=lr * muon_lr_mult, weight_decay=weight_decay),
        dict(params=other_params, use_muon=False,
             lr=lr, betas=betas, weight_decay=weight_decay),
    ]


def build_muon_optimizer(param_groups):
    """Muon + auxiliary AdamW, using the distributed implementation only under DDP."""
    import torch.distributed as dist
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

    groups = [g for g in param_groups if len(g["params"]) > 0]
    if not groups:
        raise ValueError("Muon optimizer got an empty parameter list")

    distributed = (dist.is_available() and dist.is_initialized()
                   and dist.get_world_size() > 1)
    cls = MuonWithAuxAdam if distributed else SingleDeviceMuonWithAuxAdam
    return cls(groups)


def build_optimizer(model, lr, modelconfig, tag=""):
    """Muon (default) or Adam; models without a transformer body fall back to Adam."""
    optimizer_type = modelconfig.get("optimizer", "muon")
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_type != "muon":
        raise ValueError(f"Optimizer not found: {optimizer_type}")

    groups = muon_param_groups(
        model, lr,
        muon_lr_mult=modelconfig.get("muon_lr_mult", 10.0),
        weight_decay=modelconfig.get("weight_decay", 0.01),
    )
    if groups is None:
        print(f"[optim]{tag} {type(model).__name__} has no transformer body "
              f"({MUON_BODY_MODULE_NAMES}); falling back to Adam(lr={lr})")
        return torch.optim.Adam(model.parameters(), lr=lr)

    n_muon = sum(p.numel() for p in groups[0]["params"])
    n_adam = sum(p.numel() for p in groups[1]["params"])
    print(f"[optim]{tag} Muon on {len(groups[0]['params'])} body matrices "
          f"({n_muon/1e6:.1f}M params, lr={groups[0]['lr']:.2e}) + AdamW on "
          f"{n_adam/1e6:.1f}M params (lr={groups[1]['lr']:.2e})")
    return build_muon_optimizer(groups)


def build_lr_scheduler(optimizer, trainconfig, step_size, gamma):
    """StepLR, stepped every `training.lr_decay_every_n_steps` gradient steps if set, else per epoch."""
    every_n_steps = trainconfig.get("lr_decay_every_n_steps", None)
    if every_n_steps is not None and int(every_n_steps) > 0:
        gamma = float(trainconfig.get("lr_decay_gamma", gamma))
        print(f"[optim] lr x{gamma} every {int(every_n_steps)} gradient steps")
        return {"scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=int(every_n_steps), gamma=gamma),
                "interval": "step"}
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
