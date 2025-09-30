import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from einops import repeat, rearrange
from common.fourier import isotropic_power_spectrum 

#################################################################################
#                                 PDE Losses                                    #
#################################################################################

def scaledlp_loss(input: torch.Tensor, target: torch.Tensor, p: int = 2, reduction: str = "mean"):
    B = input.size(0)
    diff_norms = torch.norm(input.reshape(B, -1) - target.reshape(B, -1), p, 1)
    target_norms = torch.norm(target.reshape(B, -1), p, 1)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)


def custommse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    loss = F.mse_loss(input, target, reduction="none")
    # avg across space
    reduced_loss = torch.mean(loss, dim=tuple(range(3, loss.ndim)))
    # sum across time + fields
    reduced_loss = reduced_loss.sum(dim=(1, 2))
    # reduce along batch
    if reduction == "mean":
        return torch.mean(reduced_loss)
    elif reduction == "sum":
        return torch.sum(reduced_loss)
    elif reduction == "none":
        return reduced_loss
    else:
        raise NotImplementedError(reduction)


def pearson_correlation(input: torch.Tensor, target: torch.Tensor, reduce_batch: bool = False):
    B = input.size(0)
    T = input.size(1)
    input = input.reshape(B, T, -1)
    target = target.reshape(B, T, -1)
    input_mean = torch.mean(input, dim=(2), keepdim=True)
    target_mean = torch.mean(target, dim=(2), keepdim=True)
    # Unbiased since we use unbiased estimates in covariance
    input_std = torch.std(input, dim=(2), unbiased=False)
    target_std = torch.std(target, dim=(2), unbiased=False)

    corr = torch.mean((input - input_mean) * (target - target_mean), dim=2) / (input_std * target_std).clamp(
        min=torch.finfo(torch.float32).tiny
    )  # shape (B, T)
    if reduce_batch:
        corr = torch.mean(corr, dim=0)
    return corr.squeeze() 


class ScaledLpLoss(torch.nn.Module):
    """Scaled Lp loss for PDEs.

    Args:
        p (int, optional): p in Lp norm. Defaults to 2.
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, p: int = 2, reduction: str = "mean") -> None:
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return scaledlp_loss(input, target, p=self.p, reduction=self.reduction)


class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return custommse_loss(input, target, reduction=self.reduction)


class PearsonCorrelationScore(torch.nn.Module):
    """Pearson Correlation Score for PDEs."""

    def __init__(self, channel: int = None, reduce_batch: bool = False) -> None:
        super().__init__()
        self.channel = channel
        self.reduce_batch = reduce_batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.channel is not None:
            input = input[:, :, self.channel]
            target = target[:, :, self.channel]
        # input in shape b nx nt or b nx ny nt
        if len(input.shape) == 3:
            input = rearrange(input, "b nx nt -> b nt nx")
            target = rearrange(target, "b nx nt -> b nt nx")
        elif len(input.shape) == 4:
            if input.shape[-1] != 1:
                input = rearrange(input, "b nx ny c -> b 1 nx ny c") # add time dimension
                target = rearrange(target, "b nx ny c -> b 1 nx ny c")
            else:
                input = rearrange(input, "b nx ny nt -> b nt nx ny") # assume time dimension is 1
                target = rearrange(target, "b nx ny nt -> b nt nx ny")
        elif len(input.shape) == 5:
            input = rearrange(input, "b nx ny nz c -> b 1 nx ny nz c") # add time dimension
            target = rearrange(target, "b nx ny nz c -> b 1 nx ny nz c")
        
        return pearson_correlation(input, target, reduce_batch=self.reduce_batch)

class KL_Loss():
    # simple VAE loss without discriminator or perceptual loss
    def __init__(self, 
                 kl_weight=1.0, 
                 criterion=nn.L1Loss()):

        super().__init__()
        self.kl_weight = kl_weight
        self.criterion = criterion

    def __call__(self, 
                inputs, 
                reconstructions, 
                posteriors,
                split="train"):
        # inputs, reconstructions in shape b nlat nlon c

        rec_loss = self.criterion(reconstructions, inputs)
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss_unweighted = kl_loss.clone().detach().mean()
        kl_loss = self.kl_weight * kl_loss

        loss = rec_loss + kl_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/kl_loss_unweighted".format(split): kl_loss_unweighted,
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
        return loss, log


def jerk_regularization_loss(pred, lat_weights=None):
    # assuming pred and gt is in shape: [batch, time, ....]
    # calculate the jerk loss, which amounts to third order temporal difference

    num_t = pred.shape[1]
    assert num_t > 3, "Number of frames must be at least 4"
    # z(t+3\delta t) - 3z(t+2\delta t) + 3z(t+\delta t) - z(t)
    diff = pred[:, 3:, ...] - 3*pred[:, 2:-1, ...] + 3*pred[:, 1:-2, ...] - pred[:, :-3, ...]

    if lat_weights is None:
        return torch.mean(diff**2)
    else:
        return torch.mean(torch.einsum('b t n ... c, n -> b t ... c', diff**2, lat_weights))

#################################################################################
#                              Climate Losses                                   #
#################################################################################

def get_cosine_weight(num_intervals, tau):
    start = 0
    end = 1
    t = np.linspace(0, 1, num_intervals+1)
    v_start = np.cos(start * np.pi / 2) ** (2 * tau)
    v_end = np.cos(end * np.pi / 2) ** (2 * tau)
    output = np.cos((t * (end - start) + start) * np.pi / 2) ** (2 * tau)
    output = 1 - (v_end - output) / (v_end - v_start)
    return output[1:]


# base on the code from graphcast
def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f'Vector {diff} is not uniformly spaced.')
    return diff[0]


def _weight_for_latitude_vector_without_poles(latitude):
    """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90 - delta_latitude/2) or
        not np.isclose(np.min(latitude), -90 + delta_latitude/2)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at '
            '+- (90 - delta_latitude/2) degrees.')
    return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
    """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90.) or
        not np.isclose(np.min(latitude), -90.)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude/2))
    # The two checks above enough to guarantee that latitudes are sorted, so
    # the extremes are the poles
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
    return weights

def disassemble_input(assembled_input, num_levels, num_surface_channels, hpx=False):
    surface_input = assembled_input[..., :num_surface_channels]
    if hpx:
        multilevel_input = rearrange(assembled_input[..., num_surface_channels:], 'b f n m (l c) -> b f n m l c', l=num_levels)
    else:
        multilevel_input = rearrange(assembled_input[..., num_surface_channels:], 'b nlat nlon (nlevel c) -> b nlat nlon nlevel c', nlevel=num_levels)
    return surface_input, multilevel_input


class WeightedLoss(nn.Module):
    def __init__(self,
                 loss_fn,
                 latitude_resolution,
                 longitude_resolution,
                 with_poles=False,
                 latitude_weight='cosine',
                 level_weight='linear',
                 multi_level_variable_weight=None,
                 surface_variable_weight=None,
                 nlevels=13,
                 nsurface=8,
                 ):
        super().__init__()
        self.loss_fn = loss_fn   # loss function must not reduce any dimension
        if latitude_weight == 'cosine':
            if with_poles:
                latitude = np.linspace(-90, 90, latitude_resolution)
                weights = _weight_for_latitude_vector_with_poles(latitude)
            else:
                # assume equiangular grid
                lat_end = (latitude_resolution-1)*(360/longitude_resolution) / 2
                latitude = np.linspace(-lat_end, lat_end, latitude_resolution)
                weights = _weight_for_latitude_vector_without_poles(latitude)
            weights = torch.from_numpy(weights)
            latitude_weight = weights / weights.mean()
        else:
            weights = torch.ones(latitude_resolution)   # all latitudes weight the same
            latitude_weight = weights / weights.mean()
        self.register_buffer('latitude_weight', latitude_weight)

        # weight for each level
        # up to 13 pressure levels, the lower the level, the lower the weight
        # the surface level has higher weight
        # 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
        if level_weight == 'linear':     # outweighs the lower levels
            level_weight = torch.linspace(0.05, 0.065, 13)
        elif level_weight == 'exp':
            level_weight = torch.exp(torch.linspace(-3, 0, 13))
            level_weight = level_weight / level_weight.sum()
        elif level_weight == 'cosine':
            level_weight = torch.from_numpy(get_cosine_weight(13, 2))
            level_weight = level_weight / level_weight.sum()
        else:
            level_weight = torch.ones(13)
            level_weight = level_weight / level_weight.sum()
        self.register_buffer('level_weight', level_weight)

        if surface_variable_weight is not None:
            surface_variable_weight = torch.tensor(surface_variable_weight)
        else:
            surface_variable_weight = torch.tensor(1.)
        self.register_buffer('surface_variable_weight', surface_variable_weight)

        if multi_level_variable_weight is not None:
            multi_level_variable_weight = torch.tensor(multi_level_variable_weight)
        else:
            multi_level_variable_weight = torch.tensor(1.)
        self.register_buffer('multi_level_variable_weight', multi_level_variable_weight)
        self.nlevels = nlevels
        self.nsurface = nsurface

    def forward(self,
                pred, # b nlat nlon (c + nlevel*c)
                target,
                ):

        surface_pred_feat, multi_level_pred_feat = disassemble_input(pred, num_levels=self.nlevels, num_surface_channels= self.nsurface)
        surface_target_feat, multi_level_target_feat = disassemble_input(target, num_levels=self.nlevels, num_surface_channels= self.nsurface)

        latitude_weight = self.latitude_weight.view(1, -1, 1) # b nlat nlon 
        surface_loss = self.loss_fn(surface_pred_feat, surface_target_feat) * self.surface_variable_weight
        surface_loss = surface_loss.sum(dim=-1) # b nlat nlon
        multi_level_loss = self.loss_fn(multi_level_pred_feat, multi_level_target_feat) * self.level_weight.view(1, 1, 1, -1, 1)
        multi_level_loss = (multi_level_loss.sum(dim=-2) * self.multi_level_variable_weight).sum(dim=-1) # b nlat nlon

        loss = surface_loss + multi_level_loss
        loss = loss * latitude_weight

        return loss.mean()   # reduce over batch/lat/lon


class LatitudeWeightedMSE(nn.Module):
    def __init__(self, nlat, nlon, loss_module=nn.MSELoss(), with_poles=False):
        super().__init__()
        self.loss_module = loss_module
        self.with_poles = with_poles
        # print(nlat, nlon)

        if not with_poles:
            longitude_resolution = nlon
            lat_end = (nlat - 1) * (360 / longitude_resolution) / 2
            lat_weight = _weight_for_latitude_vector_without_poles(np.linspace(-lat_end, lat_end, nlat))
        else:
            lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))

        lat_weight = torch.from_numpy(lat_weight)
        lat_weight = lat_weight / lat_weight.mean()
        self.register_buffer('lat_weight', lat_weight)

    def forward(self, pred, target):
        # pred, target in shape [b, nlat, nlon, c]
        lat_weight = repeat(self.lat_weight, 'nlat -> b nlat nlon', b=pred.shape[0], nlon=pred.shape[2])
        return (self.loss_module(pred, target).mean(-1) * lat_weight).mean()


def latitude_weighted_jerk_loss(pred, with_poles=False):
    # pred: [b nt nlat nlon nl c]
    longitude_resolution = pred.shape[-3]
    latitude_resolution = pred.shape[-4]
    if with_poles:
        latitude = np.linspace(-90, 90, latitude_resolution)
        weights = _weight_for_latitude_vector_with_poles(latitude)
    else:
        # assume equiangular grid
        lat_end = (latitude_resolution - 1) * (360 / longitude_resolution) / 2
        latitude = np.linspace(-lat_end, lat_end, latitude_resolution)
        weights = _weight_for_latitude_vector_without_poles(latitude)
    weights = torch.from_numpy(weights).to(pred.device).float()
    latitude_weight = weights / weights.mean()
    jerk_loss = jerk_regularization_loss(pred, lat_weights=latitude_weight)
    return jerk_loss

def latitude_weighted_rmse(pred, 
                           target,
                           with_poles=False, 
                           nlon=None,
                           nlat=None,
                           with_time=True):
    # if with_time, pred/target in shape: b t nlat nlon or b t nlat nlon l
    # else, pred/target in shape: b nlat nlon or b nlat nlon l

    if nlat is None:
        nlat = target.shape[2]
    if not with_poles:
        lat_end = (nlat-1)*(360/nlon) / 2
        lat_weight = _weight_for_latitude_vector_without_poles(np.linspace(-lat_end, lat_end, nlat))
    else:
        lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))

    lat_weight = torch.from_numpy(lat_weight).to(target.device)
    lat_weight = lat_weight / lat_weight.mean()
    if with_time:
        if len(pred.shape) == 5:
            lat_weight = lat_weight.view(1, 1, nlat, 1, 1)
        else:
            lat_weight = lat_weight.view(1, 1, nlat, 1)
        return torch.sqrt((((pred - target)**2) * lat_weight).mean(dim=(2, 3)))   # spatial averaging
    else:
        if len(pred.shape) == 4:
            lat_weight = lat_weight.view(1, nlat, 1, 1)
        else:
            lat_weight = lat_weight.view(1, nlat, 1)
        return torch.sqrt((((pred - target)**2) * lat_weight).mean(dim=(1, 2)))   # spatial averaging

def rmse(pred, target):
    # directly infer latitude from target: b t nface nside nside or b t nface nside nside l
    return torch.sqrt(((pred - target)**2).mean(dim=(2, 3, 4)))   # spatial averaging

def latitude_weighted_l1(pred, target):
    # directly infer latitude from target: b t nlat nlon or b t nlat nlon l
    nlat = target.shape[2]
    lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))
    lat_weight = torch.from_numpy(lat_weight).to(target.device)
    lat_weight = lat_weight / lat_weight.mean()
    if len(pred.shape) == 5:
        lat_weight = lat_weight.view(1, 1, nlat, 1, 1)
    else:
        lat_weight = lat_weight.view(1, 1, nlat, 1)

    return ((pred - target).abs() * lat_weight).mean(dim=(2, 3))   # spatial averaging

class CRPS(nn.Module):
    def __init__(self, nlat, nlon, with_poles=False):
        super().__init__()
        self.with_poles = with_poles

        if not with_poles:
            longitude_resolution = nlon
            lat_end = (nlat - 1) * (360 / longitude_resolution) / 2
            lat_weight = _weight_for_latitude_vector_without_poles(np.linspace(-lat_end, lat_end, nlat))
        else:
            lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))

        lat_weight = torch.from_numpy(lat_weight)
        lat_weight = lat_weight / lat_weight.mean()
        self.register_buffer('lat_weight', lat_weight) # shape [nlat]

    def crps_skill(self, pred, target):
        # pred in shape [b, ens, nt, nlat, nlon]
        # target in shape [b, nt, nlat, nlon]
        b = pred.shape[0]
        nt = pred.shape[2]
        nlon = pred.shape[4]

        skill = torch.abs(pred - target.unsqueeze(1)) # b ens nt nlat nlon
        skill = skill.mean(dim=1) # b nt nlat nlon
        lat_weight = repeat(self.lat_weight, 'nlat -> b nt nlat nlon', b=b, nt=nt, nlon=nlon) # b nt nlat nlon
        avg_skill = skill * lat_weight # b nt nlat nlon
        avg_skill = avg_skill.mean(dim=(2, 3)) # b nt

        return avg_skill    
    
    def crps_spread(self, pred, target):
        # pred in shape [b, ens, nt, nlat, nlon]
        # target in shape [b, nt, nlat, nlon]        
        b, ens, nt, nlat, nlon = pred.shape

        prefactor = 1/(ens*(ens-1))

        pred_diffs = torch.zeros((b, ens, ens, nt, nlat, nlon), device=pred.device) # b ens ens nt nlat nlon c
        for i in range(ens):
            for j in range(ens):
                pred_diffs[:, i, j] = torch.abs(pred[:, i] - pred[:, j])

        pred_diffs = pred_diffs.sum(dim=(1, 2)) * prefactor  # b nt nlat nlon

        lat_weight = repeat(self.lat_weight, 'nlat -> b nt nlat nlon', b=b, nt=nt, nlon=nlon) # b nt nlat nlon
        avg_spread = pred_diffs * lat_weight # b nt nlat nlon
        avg_spread = avg_spread.mean(dim=(2, 3)) # b nt

        return avg_spread
    
    def forward(self, pred, target):
        # pred in shape [b, ens, nt, nlat, nlon]
        # target in shape [b, nt, nlat, nlon]
        skill = self.crps_skill(pred, target)  # b nt
        spread = self.crps_spread(pred, target)  # b nt

        crps = skill - 0.5*spread # b nt
        
        crps_avg = torch.zeros_like(crps) # b nt
        t = crps.shape[1]

        for i in range(t):
            crps_avg[:, i] = crps[:, :i+1].mean(dim=1)
        
        return crps_avg
    
def sRMSE(pred, target, spatial=2):
    # u, v: [b, nx, ny, d]
    # u is prediction, v is target
    sRMSE_all = []
    for i in range(pred.shape[-1]):
        u = pred[..., i]
        v = target[..., i]
        p_u, k = isotropic_power_spectrum(u, spatial=spatial)
        p_v, _ = isotropic_power_spectrum(v, spatial=spatial)
        p_u = p_u.mean(dim=0)
        p_v = p_v.mean(dim=0)

        se_p = torch.square(1 - (p_v + 1e-6) / (p_u + 1e-6))

        rmse_f = []

        bins = torch.logspace(k[0].log2(), -1.0, steps=4, base=2)

        for i in range(4):
            if i < 3:
                mask = torch.logical_and(bins[i] <= k, k <= bins[i + 1])
            else:
                mask = bins[i] <= k

            rmse_f.append(torch.sqrt(torch.mean(se_p[mask])))

        sRMSE_all.append(torch.stack(rmse_f))
    
    sRMSE_all = torch.stack(sRMSE_all, dim=0)  # d x 4
    sRMSE_all = torch.mean(sRMSE_all, dim=0)  # 4
    
    return sRMSE_all

def VRMSE(pred, target):
    # pred, target in shape b nx ny d
    v = pred
    u = target 
    se = torch.square(u - v) # b nx ny d
    mse = torch.mean(se) # ()
    vrmse = torch.sqrt(mse / (torch.var(u) + 1e-6)) # ()
    return vrmse