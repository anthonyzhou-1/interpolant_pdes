import lightning as L
import time
import os
import torch
import numpy as np
import pickle 
from einops import rearrange

from modules.models.FNO2D import FNO2d
from modules.models.DiT import DIT
from modules.models.ClimaDiT import ClimaDIT

from modules.diffusion.base import DiffusionScheduler
from modules.diffusion.flow_matching import LinearScheduler
from modules.diffusion.edm import EDMScheduler
from modules.diffusion.interpolant import DriftScheduler

from common.utils import build_lr_scheduler, build_optimizer
from common.loss import ScaledLpLoss, PearsonCorrelationScore, latitude_weighted_rmse, sRMSE, VRMSE
from common.loss import ensemble_crps, ensemble_ssr, latitude_weight
from common.plotting import plot_result_2d, plot_entropy_2d
from common.climate_utils import plot_result_climate, plot_spectrum
from common.climate_utils import assemble_scalar_params, assemble_grid_params, assemble_input, disassemble_input
from dataset.plasim import SURFACE_FEATURES, MULTI_LEVEL_FEATURES

# Climate variables for validation metrics: (log label, feature key, level index or None)
CLIMATE_TRACKED_VARS = (("t2m", "tas", None),
                        ("pr_6h", "pr_6h", None),
                        ("z500", "zg", 7),
                        ("u250", "ua", 4),
                        ("t850", "ta", 10))

# lead time in hours -> index into the rollout, at 6 h per step
CLIMATE_LEAD_TIMES = ((6, 0), (24, 3), (72, 11), (120, 19), (240, 39))


class TrainModule(L.LightningModule):
    def __init__(self,
                 config: dict,
                 normalizer= None):
        '''
        TrainModule
        args:
            config (dict): configuration dictionary containing model, training and data configurations
            normalizer (object, optional): normalizer object for scaling input data. Defaults to None.
        '''

        super().__init__()
        self.config=config
        self.modelconfig = config['model']
        self.model_name = self.modelconfig["model_name"]
        self.lr = self.modelconfig["lr"]
        self.log_dir = config['training']['log_dir']
        self.correlation = 0.8
        self.pde = config['data']['pde']
        self.eval_all = config['training'].get('eval_all', True) # plot all validation batches, not just the first
        # plot_val=False disables validation plotting and pickling entirely
        self.plot_val = config['training'].get('plot_val', True)
        self.normalizer = normalizer
       
        self.criterion = ScaledLpLoss()
        self.correlation_criterion = PearsonCorrelationScore(reduce_batch=True)

        # flags for using probabilistic models or latent space models
        self.diffusion = False
        self.latent = self.modelconfig.get("latent", False)

        # sampler wall-clock timing (enabled by eval_nfe.py)
        self.time_sampling = False
        self._sample_time_s = 0.0
        self._sample_calls = 0
        self._sample_members = 0

        # Validation ensemble size. Deterministic metrics use the member mean; CRPS/SSR use the members.
        self.ensemble_size = max(1, int(config['training'].get('val_ensemble_size', 1) or 1))

        # Per-timestep metric curves, written to log_dir by on_validation_epoch_end.
        # val_tag distinguishes validation runs that share a log_dir.
        self.val_tag = ""
        self._val_curves = {}
        self._val_curve_counts = {}
        self._lat_weight = None

        # number of steps dropped due to non-finite gradients
        self._nonfinite_steps = 0

        if self.model_name == "fno2d":
            fnoconfig = self.modelconfig["fno2d"]
            self.model = FNO2d(**fnoconfig)
            self.latent = False 
        elif self.model_name == "sfno":
            assert self.pde == "climate"
            try:
                from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperatorNet as SFNO
                from common.spherical_loss import L2LossS2
            except ImportError as e:
                raise ImportError(
                    "sfno needs torch_harmonics >= 0.7.4 built against the installed "
                    f"torch; importing it failed with: {e}") from e
            
            sfno_config = self.modelconfig["sfno"]
            self.model = SFNO(**sfno_config)
            self.latent = False
            self.criterion = L2LossS2(nlat=64, nlon=128)

        # DDPM, DDIM, TSM share same training, but different sampling
        elif self.model_name == "ddpm" or self.model_name == "ddim" or self.model_name == "tsm":
            self.diffusion = True 
            diffusionconfig = self.modelconfig[self.model_name]
            ditconfig = self.modelconfig["dit"]
            if self.pde == "climate":
                self.model = ClimaDIT(config)
            else:
                self.model = DIT(**ditconfig)
            self.scheduler = DiffusionScheduler(mode=self.model_name,
                                                **diffusionconfig)
                
        elif self.model_name == "edm":
            self.diffusion = True
            edmconfig = self.modelconfig["edm"]
            ditconfig = self.modelconfig["dit"]
            if self.pde == "climate":
                self.model = ClimaDIT(config)
            else:
                self.model = DIT(**ditconfig)
            self.scheduler = EDMScheduler(**edmconfig)

        elif self.model_name == "flow_matching":
            self.diffusion = True
            fmconfig = self.modelconfig["flow_matching"]
            ditconfig = self.modelconfig["dit"]
            if self.pde == "climate":
                self.model = ClimaDIT(config)
            else:
                self.model = DIT(**ditconfig)
            self.scheduler = LinearScheduler(**fmconfig)

        elif self.model_name == "interpolant":
            self.diffusion = True
            interpolantconfig = self.modelconfig["interpolant"]
            ditconfig = self.modelconfig["dit"]
            if self.pde == "climate":
                self.model = ClimaDIT(config)
            else:
                self.model = DIT(**ditconfig)
            self.scheduler = DriftScheduler(**interpolantconfig)

        else:
            self.model = None
            raise NotImplementedError(f"Model {self.model_name} not implemented")

        if self.latent:
            self.init_autoencoder()

        if config['training']['strategy'] == 'ddp' or config['training']['strategy'] == 'ddp_find_unused_parameters_true':
            self.ddp = True
        else:
            self.ddp = False

        self.save_hyperparameters()
    
    def init_autoencoder(self):
        from modules.ae_module import AutoencoderModule
        aeconfig = self.modelconfig['autoencoder']
        checkpoint = torch.load(aeconfig['checkpoint'], map_location='cpu', weights_only=False)
        self.autoencoder = AutoencoderModule(aeconfig, normalizer=self.normalizer)
        self.autoencoder.load_state_dict(checkpoint['state_dict'])
        
        # freeze the autoencoder
        self.autoencoder = self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.scale_factor = aeconfig.get('scale_factor', 1.0)

    @property
    def effective_nfe(self):
        """
        Number of network evaluations used to generate one frame.
        """
        if not self.diffusion:
            return 1
        if self.model_name == "ddpm":
            return int(self.scheduler.noise_steps)
        if self.model_name == "tsm":
            return int(round(self.scheduler.noise_steps * (1.0 - self.scheduler.skip_percent)))
        if self.model_name == "ddim":
            return int(self.scheduler.num_ddim_steps)
        if self.model_name == "edm":
            # heun takes a 2nd-order correction step at every step but the last
            if getattr(self.scheduler, "solver", "euler") == "heun":
                return int(2 * self.scheduler.num_steps - 1)
            return int(self.scheduler.num_steps)
        if self.model_name in ("flow_matching", "interpolant"):
            return int(self.scheduler.num_refinement_steps)
        raise NotImplementedError(f"effective_nfe undefined for {self.model_name}")

    def set_nfe(self, n):
        """
        Set the sampling budget for methods whose NFE is a free parameter.

        No-op for ddpm (always runs its full reverse chain) and for deterministic models (NFE 1).
        Returns the effective NFE after the change, which may differ from n.
        """
        if self.diffusion:
            if self.model_name in ("flow_matching", "interpolant"):
                self.scheduler.num_refinement_steps = int(n)
            elif self.model_name == "edm":
                self.scheduler.num_steps = int(n)
                self.scheduler.noise_steps = int(n)  # churn strength reads this
            elif self.model_name == "ddim":
                self.scheduler.num_ddim_steps = int(n)
                self.scheduler.ref_arr = None  # force setup_ddim_sampling() to rerun
            elif self.model_name == "tsm":
                # stop the reverse chain after n of noise_steps steps
                steps = self.scheduler.noise_steps
                self.scheduler.skip_percent = max(0.0, 1.0 - float(n) / steps)
        return self.effective_nfe

    # models whose sampler has both a deterministic and a stochastic scheme
    VARIANT_MODELS = ("edm", "interpolant")

    @property
    def sampler_variant(self):
        """Label for the sampling scheme, where the model has more than one."""
        if self.model_name == "edm":
            return ("sde" if self.scheduler.stochastic else "ode") + f"-{self.scheduler.solver}"
        if self.model_name == "interpolant":
            return "sde" if self.scheduler.method == "em" else "ode"
        return "default"

    @property
    def sampler_sigma(self):
        """
        Noise scale injected by the interpolant sampler (0 for the euler integrator).
        """
        if self.model_name != "interpolant":
            return float("nan")
        return float(self.scheduler.sigma_sample) if self.scheduler.method == "em" else 0.0

    def set_sampler_variant(self, variant, sigma=None):
        """
        Select a sampling scheme: 'ode' (deterministic) or 'sde' (stochastic), optionally
        suffixed with a solver for edm, e.g. 'sde-heun'. Returns the resulting label.

        edm        : ode = probability-flow ODE, sde = Karras churn.
        interpolant: ode = euler integrator, sde = Euler-Maruyama at `sigma`
                     (defaults to the trained sigma_coef).
        """
        if variant is None or variant == "default":
            return self.sampler_variant
        if self.model_name not in self.VARIANT_MODELS:
            raise ValueError(f"{self.model_name} has no sampler variants (got {variant!r})")
        kind, _, solver = variant.partition("-")
        if kind not in ("ode", "sde"):
            raise ValueError(f"variant must be ode or sde, got {kind!r}")

        if self.model_name == "edm":
            self.scheduler.stochastic = (kind == "sde")
            if solver:
                assert solver in ("euler", "heun"), f"unknown edm solver: {solver}"
                self.scheduler.solver = solver
        else:  # interpolant
            assert not solver, f"interpolant takes no solver suffix (got {variant!r})"
            method = "em" if kind == "sde" else "euler"
            self.scheduler.method = method
            self.scheduler.integrator.method = method   # Integrator.step_fn reads its own copy
            if kind == "sde" and sigma is not None:
                self.scheduler.sigma_sample = float(sigma)
        return self.sampler_variant

    @property
    def stochastic_sampling(self):
        """
        Whether two samples from the same conditioning differ (interpolant with
        integrator='euler' is deterministic).
        """
        if not self.diffusion:
            return False
        if self.model_name == "interpolant":
            return self.scheduler.method == "em"
        return True

    def set_ensemble(self, n):
        """Set the number of ensemble members used by validation_step. Returns the value set."""
        self.ensemble_size = max(1, int(n))
        return self.ensemble_size

    def reset_timing(self):
        self._sample_time_s = 0.0
        self._sample_calls = 0
        self._sample_members = 0

    def _timed_forward(self, *args, **kwargs):
        """forward(), optionally wrapped in a synchronized timer."""
        if not self.time_sampling:
            return self.forward(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.forward(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._sample_time_s += time.perf_counter() - t0
        self._sample_calls += 1
        # count members (b * ensemble_size) for per-member timing
        try:
            self._sample_members += int(args[0].shape[0])
        except (IndexError, AttributeError):
            pass
        return out

    def forward(self, u, cond=None, scalar_params=None, grid_params=None):
        # generative models
        if self.diffusion:
            if self.pde == "climate":
                return self.scheduler.sample(u, self.model,
                                             scalar_params=scalar_params, 
                                             grid_params=grid_params)
            else:
                return self.scheduler.sample(u, self.model, cond=cond)
        # deterministic models
        else:
            if self.pde == "climate":
                scalar_params = scalar_params.view(-1, 1, 1, scalar_params.shape[-1]).repeat(1, u.shape[1], u.shape[2], 1) # b nlat nlon c
                u = torch.cat((u, scalar_params, grid_params), dim=-1) # b nlat nlon c + c + l*c
                u = rearrange(u, 'b h w c -> b c h w')
                pred = self.model(u)
                return rearrange(pred, 'b c h w -> b h w c')
            else:
                if self.model_name == "lns":
                    return self.model(u, None, cond)
                else:
                    return self.model(u, cond)
    
    def encode(self, u, cond=None):
        """
        Encode input to latent space using the autoencoder.
        Args:
            u_input (torch.Tensor): Input tensor to be encoded.
        Returns:
            torch.Tensor: Encoded latent representation.
        """
        if self.latent is False:
            return u

        with torch.no_grad():
            z, posterior = self.autoencoder.encode(u, cond)
            z = z * self.scale_factor  # scale the latent representation
            if len(z.shape) == 4:
                z = z.permute(0, 2, 3, 1)  # b c nx ny -> b nx ny c
            elif len(z.shape) == 5:
                z = z.permute(0, 2, 3, 4, 1)
        return z
    
    def decode(self, z, cond=None):
        '''
        Decode latent representation back to input space using the autoencoder.
        Args:
            z (torch.Tensor): Latent representation to be decoded.
        Returns:
            torch.Tensor: Decoded input tensor.
        '''
        if self.latent is False:
            return z

        with torch.no_grad():
            if len(z.shape) == 4:
                z = z.permute(0, 3, 1, 2)
            elif len(z.shape) == 5:
                z = z.permute(0, 4, 1, 2, 3) # b nx ny nz c -> b c nx ny nz
            u = self.autoencoder.decode(z / self.scale_factor, cond)
        return u
    
    def get_data(self, batch, val=False):
        u_label = batch["output_fields"]

        cond = batch.get("constant_scalars", None) # b num_cond
        if self.pde == "rayleigh_benard":
            cond[:, 0] = torch.log10(cond[:, 0]) # log Rayleigh number

        if val: # u_label in shape (b, nt, nx, ny, c)
            u_input = u_label[:, 0] # get first step of trajectory
            return u_input, u_label, cond

        u_input = batch["input_fields"] # b nx ny c 

        # the_well inserts a time dimension = 1
        if len(u_input.shape) > 4: # b 1 nx ny c
            u_input = u_input[:, 0] # b nx ny c
            u_label = u_label[:, 0] # b nx ny c

        return u_input, u_label, cond
    
    def training_step(self, batch, batch_idx):
        if self.pde == "climate":
            surface_feat, multi_level_feat, constants, yearly_constants, day_of_year, hour_of_day = batch  
            scalar_params = assemble_scalar_params(day_of_year, hour_of_day, 0) # b 2
            grid_params = assemble_grid_params(constants, yearly_constants, 0) # b nlat nlon (c + c)

            u_input = assemble_input(surface_feat[:, 0], multi_level_feat[:, 0])
            u_target = assemble_input(surface_feat[:, 1], multi_level_feat[:, 1])

            if self.latent:
                u_input = self.encode(u_input, None) # b zlat zlon z
                u_target = self.encode(u_target, None) # b zlat zlon z

            if self.diffusion:
                loss = self.scheduler.compute_loss(u_input, u_target, self.model, scalar_params=scalar_params, grid_params=grid_params)
            else:
                scalar_params = scalar_params.view(-1, 1, 1, scalar_params.shape[-1]).repeat(1, u_input.shape[1], u_input.shape[2], 1)
                u_input = torch.cat((u_input, scalar_params, grid_params), dim=-1) # b nlat nlon c + c + l*c
                u_input = rearrange(u_input, 'b h w c -> b c h w')
                u_pred = self.model(u_input)
                u_target = rearrange(u_target, 'b h w c -> b c h w')
                loss = self.criterion(u_pred, u_target)

            self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)

        else:
            u_input, u_label, cond = self.get_data(batch)

            if self.latent:
                u_input = self.encode(u_input, cond)
                u_label = self.encode(u_label, cond)

            if self.diffusion:
                loss = self.scheduler.compute_loss(u_input, u_label, self.model, cond=cond)
            else:
                u_pred = self.forward(u_input, cond)
                loss = self.criterion(u_pred, u_label)
            self.log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Zero the gradients of a batch with non-finite gradients, since a single NaN
        would otherwise poison the optimizer state for the rest of training."""
        flags = [torch.isfinite(p.grad).all()
                 for group in optimizer.param_groups
                 for p in group["params"] if p.grad is not None]
        if flags and not bool(torch.stack(flags).all()):
            self._nonfinite_steps += 1
            print(f"Non-finite gradient at global step {self.global_step}; dropping this "
                  f"batch's update ({self._nonfinite_steps} dropped so far).")
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.zero_()

        self.log('train/nonfinite_steps', float(self._nonfinite_steps),
                 on_step=True, on_epoch=False, sync_dist=self.ddp)

    def validation_step(self, batch, batch_idx, eval=False, z_pred=None, ensemble_size=1, return_ens=False):

        # an explicit ensemble_size takes precedence over set_ensemble()
        if ensemble_size == 1:
            ensemble_size = self.ensemble_size

        # deterministic samplers would produce identical members
        if ensemble_size > 1 and not self.stochastic_sampling:
            ensemble_size = 1

        if self.pde == "climate":
            surface_feat, multi_level_feat, constants, yearly_constants, day_of_year, hour_of_day = batch    
        
            loss_dict, pred_feat_dict, target_feat_dict, z_pred, pred_ens_feat_dict = self.predict_climate(
                                                            surface_feat, 
                                                            multi_level_feat,
                                                            day_of_year,
                                                            hour_of_day,
                                                            constants,
                                                            yearly_constants,
                                                            return_pred=True,
                                                            z_pred=z_pred,
                                                            ensemble_size=ensemble_size,
                                                            return_ens=return_ens)
            
            if eval:
                return loss_dict, pred_feat_dict, target_feat_dict, z_pred

            # visualize the prediction for first batch
            if self.plot_val and batch_idx == 0:
                if self.ddp and self.global_rank != 0:
                    pass
                else: # not ddp or rank 0
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
                                    f'{self.log_dir}/val_t2m_{self.global_step}.png')
                    plot_result_climate(z500_pred,
                                    z500_target,
                                    f'{self.log_dir}/val_z500_{self.global_step}.png')
                    plot_result_climate(pr_6h_pred,
                                    pr_6h_target,
                                    f'{self.log_dir}/val_pr_6h_{self.global_step}.png')
                    plot_result_climate(u250_pred,
                                    u250_target,
                                    f'{self.log_dir}/val_u250_{self.global_step}.png')
                    plot_result_climate(t850_pred,
                                    t850_target,
                                    f'{self.log_dir}/val_t850_{self.global_step}.png')
                    
                    plot_spectrum(t2m_pred,
                                    t2m_target,
                                    f'{self.log_dir}/val_t2m_spectrum_{self.global_step}.png')
                    plot_spectrum(z500_pred,
                                    z500_target,
                                    f'{self.log_dir}/val_z500_spectrum_{self.global_step}.png')
                    plot_spectrum(pr_6h_pred,
                                    pr_6h_target,
                                    f'{self.log_dir}/val_pr_6h_spectrum_{self.global_step}.png')
                    plot_spectrum(u250_pred,
                                    u250_target,
                                    f'{self.log_dir}/val_u250_spectrum_{self.global_step}.png')
                    plot_spectrum(t850_pred,
                                    t850_target,
                                    f'{self.log_dir}/val_t850_spectrum_{self.global_step}.png')
            
            # calculate the mean loss, shape b t for each key, b t l for multilevel keys
            t2m_loss = loss_dict['tas'].mean(0) # surface temp, mean across batch dim
            pr_6h_loss = loss_dict['pr_6h'].mean(0) # 6-hour accumulated precipitation
            z500_loss = loss_dict['zg'][..., 7].mean(0) # geopotential at level=7
            u250_loss = loss_dict['ua'][..., 4].mean(0) # u wind at level=4
            t850_loss = loss_dict['ta'][..., 10].mean(0) # temp at level=10
            
            self.log('val/t2m_6', t2m_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 6 hours
            self.log('val/t2m_24', t2m_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 1 day
            self.log('val/t2m_72', t2m_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 3 day
            self.log('val/t2m_120', t2m_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 5 day
            self.log('val/t2m_240', t2m_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 10 day

            self.log('val/pr_6h_6', pr_6h_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/pr_6h_24', pr_6h_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/pr_6h_72', pr_6h_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/pr_6h_120', pr_6h_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/pr_6h_240', pr_6h_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

            self.log('val/z500_6', z500_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/z500_24', z500_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/z500_72', z500_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/z500_120', z500_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/z500_240', z500_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

            self.log('val/u250_6', u250_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/u250_24', u250_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/u250_72', u250_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/u250_120', u250_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/u250_240', u250_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

            self.log('val/t850_6', t850_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/t850_24', t850_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/t850_72', t850_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/t850_120', t850_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/t850_240', t850_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

            # probabilistic metrics (latitude-weighted CRPS and SSR)
            target_t2m = target_feat_dict['tas']            # b t nlat nlon
            n_samples, nt_val, nlat, nlon = target_t2m.shape
            lat_w = self._climate_lat_weight(nlat, nlon, target_t2m)
            ens = pred_ens_feat_dict['tas'].shape[1]

            for label, key, level in CLIMATE_TRACKED_VARS:
                pred_ens = pred_ens_feat_dict[key]
                target_var = target_feat_dict[key]
                if level is not None:
                    pred_ens = pred_ens[..., level]
                    target_var = target_var[..., level]
                crps_t = ensemble_crps(pred_ens, target_var, spatial_dims=(-2, -1), weight=lat_w).mean(0)
                ssr_t = ensemble_ssr(pred_ens, target_var, spatial_dims=(-2, -1), weight=lat_w).mean(0)

                self.log(f'val/crps_{label}', crps_t.mean().item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
                self.log(f'val/ssr_{label}', ssr_t.mean().item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
                for hours, idx in CLIMATE_LEAD_TIMES:
                    if idx >= crps_t.numel():
                        continue
                    self.log(f'val/crps_{label}_{hours}', crps_t[idx].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
                    self.log(f'val/ssr_{label}_{hours}', ssr_t[idx].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

                self._accumulate_curve(f'crps_{label}', crps_t.tolist(), n_samples, nt_val)
                self._accumulate_curve(f'ssr_{label}', ssr_t.tolist(), n_samples, nt_val)

            self.log('val/ensemble_size', float(ens), on_step=False, on_epoch=True, sync_dist=self.ddp)

            for label, curve in (('rmse_t2m', t2m_loss), ('rmse_pr_6h', pr_6h_loss),
                                 ('rmse_z500', z500_loss), ('rmse_u250', u250_loss),
                                 ('rmse_t850', t850_loss)):
                self._accumulate_curve(label, curve.tolist(), n_samples, nt_val)

        else:
            u_input, u_label, cond = self.get_data(batch, val=True)

            u_pred_denorm = torch.zeros_like(u_label) # shape (b, nt, nx, ny, c)
            u_pred_denorm[:, 0] =  self.normalizer.denormalize(u_input) # save initial condition

            nt = u_label.shape[1]
            accumulated_loss = []
            sRMSE_low = []
            sRMSE_mid = []
            sRMSE_high = []
            sRMSE_total = []
            accumulated_VRMSE = []
            accumulated_CRPS = []
            accumulated_SSR = []
            # single-member scores (only for ens > 1)
            accumulated_VRMSE_single = []
            sRMSE_low_single, sRMSE_mid_single, sRMSE_high_single = [], [], []
            at_correlation = False

            if self.latent:
                u_input = self.encode(u_input, cond) # encode input to latent space

            # roll out E members independently from the same initial condition
            ens = ensemble_size
            cond_ens = cond
            if ens > 1:
                u_input = u_input.repeat_interleave(ens, dim=0) # (b*ens) ...
                if cond is not None:
                    cond_ens = cond.repeat_interleave(ens, dim=0)

            for i in range(0, nt-1):
                pred = self._timed_forward(u_input, cond_ens) # shape (b*ens, nx, ny, 1)
                u_true = u_label[:, i+1]

                if pred.isnan().any():
                    print(f"NaN detected in prediction at step {i+1}.")
                    break

                true_denorm = self.normalizer.denormalize(u_true)
                if self.latent:
                    pred_u = self.decode(pred, cond_ens) # decode prediction from latent space
                else:
                    pred_u = pred

                pred_denorm = self.normalizer.denormalize(pred_u)

                # b ens ...
                members = pred_denorm.reshape(-1, ens, *pred_denorm.shape[1:])

                accumulated_CRPS.append(ensemble_crps(members, true_denorm).mean().item())
                accumulated_SSR.append(ensemble_ssr(members, true_denorm).mean().item())

                # the deterministic metrics are scored on the ensemble mean
                pred_denorm = members.mean(1)

                u_pred_denorm[:, i+1] = pred_denorm # save prediction

                loss = self.criterion(pred_denorm, true_denorm) # calculate loss
                accumulated_loss.append(loss.item())

                vrmse = VRMSE(pred_denorm, true_denorm)
                accumulated_VRMSE.append(vrmse.item())

                correlation = self.correlation_criterion(pred_denorm, true_denorm) # calculate correlation
                if correlation < self.correlation and not at_correlation:
                    correlation_time = float(i+1) # get time step at correlation
                    at_correlation = True 

                if len(u_input.shape) > 4:
                    spatial = 3
                else:
                    spatial = 2

                sRMSE_all = sRMSE(pred_denorm, true_denorm, spatial=spatial) # calculate sRMSE
                sRMSE_low.append(sRMSE_all[0].item())
                sRMSE_mid.append(sRMSE_all[1].item())
                sRMSE_high.append(sRMSE_all[2].item())
                sRMSE_total.append(sRMSE_all[3].item())

                if ens > 1:
                    single = members[:, 0]
                    accumulated_VRMSE_single.append(VRMSE(single, true_denorm).item())
                    s1 = sRMSE(single, true_denorm, spatial=spatial)
                    sRMSE_low_single.append(s1[0].item())
                    sRMSE_mid_single.append(s1[1].item())
                    sRMSE_high_single.append(s1[2].item())

                u_input = pred # update input for next step (per member, not the mean)

            # rollout diverged on the first step: log metrics as inf
            if not accumulated_loss:
                print(f"Rollout produced no usable steps (batch {batch_idx}, global step "
                      f"{self.global_step}); logging this batch's val metrics as inf.")
                inf = float('inf')
                for i in range(10):
                    self.log(f'val/VRMSE_{i}', inf, on_step=False, on_epoch=True, sync_dist=self.ddp)
                keys = ['val/loss', 'val/VRMSE', 'val/sRMSE_low', 'val/sRMSE_mid',
                        'val/sRMSE_high', 'val/sRMSE_total', 'val/CRPS']
                if ens > 1:
                    keys += ['val/VRMSE_single', 'val/sRMSE_low_single',
                             'val/sRMSE_mid_single', 'val/sRMSE_high_single']
                for key in keys:
                    self.log(key, inf, on_step=False, on_epoch=True, sync_dist=self.ddp)
                self.log('val/SSR', 0.0, on_step=False, on_epoch=True, sync_dist=self.ddp)
                self.log('val/correlation_time', 0.0, on_step=False, on_epoch=True, sync_dist=self.ddp)
                return

            if not at_correlation:
                correlation_time = nt-1 # didn't go below correlation threshold, therefore the time is the last step

            loss = sum(accumulated_loss) / len(accumulated_loss)
            VRMSE_total = sum(accumulated_VRMSE) / len(accumulated_VRMSE)

            len_rollout = len(accumulated_VRMSE)
            for i in range(10):
                start = i * (len_rollout // 10)
                end = (i + 1) * (len_rollout // 10)
                if end <= start:
                    continue
                rollout_loss_i = sum(accumulated_VRMSE[start:end]) / (end - start)
                self.log(f'val/VRMSE_{i}', rollout_loss_i, on_step=False, on_epoch=True, sync_dist=self.ddp)

            self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/correlation_time', correlation_time, on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/VRMSE', VRMSE_total, on_step=False, on_epoch=True, sync_dist=self.ddp)

            self.log('val/sRMSE_low', sum(sRMSE_low) / len(sRMSE_low), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/sRMSE_mid', sum(sRMSE_mid) / len(sRMSE_mid), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/sRMSE_high', sum(sRMSE_high) / len(sRMSE_high), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/sRMSE_total', sum(sRMSE_total) / len(sRMSE_total), on_step=False, on_epoch=True, sync_dist=self.ddp)

            self.log('val/CRPS', sum(accumulated_CRPS) / len(accumulated_CRPS),
                     on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/SSR', sum(accumulated_SSR) / len(accumulated_SSR),
                     on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/ensemble_size', float(ens), on_step=False, on_epoch=True, sync_dist=self.ddp)

            # single-member metrics (ens > 1 only)
            for key, curve in (('val/VRMSE_single', accumulated_VRMSE_single),
                               ('val/sRMSE_low_single', sRMSE_low_single),
                               ('val/sRMSE_mid_single', sRMSE_mid_single),
                               ('val/sRMSE_high_single', sRMSE_high_single)):
                if curve:
                    self.log(key, sum(curve) / len(curve),
                             on_step=False, on_epoch=True, sync_dist=self.ddp)

            n_samples = u_label.shape[0]
            for name, curve in (('loss', accumulated_loss),
                                ('VRMSE', accumulated_VRMSE),
                                ('CRPS', accumulated_CRPS),
                                ('SSR', accumulated_SSR),
                                ('VRMSE_single', accumulated_VRMSE_single),
                                ('sRMSE_low', sRMSE_low),
                                ('sRMSE_mid', sRMSE_mid),
                                ('sRMSE_high', sRMSE_high),
                                ('sRMSE_total', sRMSE_total)):
                self._accumulate_curve(name, curve, n_samples, nt - 1)

            if self.plot_val and (batch_idx == 0 or self.eval_all): # only plot first batch or if eval_all flag is set
                if self.ddp and self.global_rank != 0:
                    pass
                else:
                    u_label_denorm = self.normalizer.denormalize(u_label)

                    if len(u_label_denorm.shape) > 5: # b nt nx ny nz c
                        u_label_denorm = u_label_denorm[:, :, 0] 
                        u_pred_denorm = u_pred_denorm[:, :, 0] 

                    plot_result_2d(u_label_denorm, u_pred_denorm, n_t=10, path=f'{self.log_dir}batch_{batch_idx}_step_{self.global_step}.png')
                    if self.pde == "km_flow":
                        plot_entropy_2d(u_label_denorm, u_pred_denorm, n_t=5, path=f'{self.log_dir}batch_{batch_idx}_entropy_step_{self.global_step}.png')
                    
                    with open(f'{self.log_dir}accumulated_loss_batch_{batch_idx}_step{self.global_step}.pkl', 'wb') as f:
                        pickle.dump(accumulated_loss, f)

                    with open(f'{self.log_dir}accumulated_VRMSE_batch_{batch_idx}_step{self.global_step}.pkl', 'wb') as f:
                        pickle.dump(accumulated_VRMSE, f)

                    with open(f'{self.log_dir}sRMSE_low_batch_{batch_idx}_step{self.global_step}.pkl', 'wb') as f:
                        pickle.dump(sRMSE_low, f)

                    with open(f'{self.log_dir}sRMSE_mid_batch_{batch_idx}_step{self.global_step}.pkl', 'wb') as f:
                        pickle.dump(sRMSE_mid, f)

                    with open(f'{self.log_dir}sRMSE_high_batch_{batch_idx}_step{self.global_step}.pkl', 'wb') as f:
                        pickle.dump(sRMSE_high, f)
    
    def _climate_lat_weight(self, nlat, nlon, like):
        '''Cosine-latitude weights shaped to broadcast over a (..., nlat, nlon) field.'''
        if self._lat_weight is None or self._lat_weight.shape[0] != nlat:
            self._lat_weight = latitude_weight(nlat, nlon,
                                               with_poles=self.config["data"]["with_poles"]).view(nlat, 1)
        return self._lat_weight.to(device=like.device, dtype=like.dtype)

    def on_validation_epoch_start(self):
        self._val_curves = {}
        self._val_curve_counts = {}

    def _accumulate_curve(self, name, values, n_samples, length):
        '''
        Add one batch's metric-vs-lead-time curve to the running validation-set sum.
        Curves are zero-padded to `length` with per-timestep sample counts.
        '''
        k = min(len(values), length)
        if k == 0:
            return
        total = torch.zeros(length, dtype=torch.float64)
        count = torch.zeros(length, dtype=torch.float64)
        total[:k] = torch.as_tensor(values[:k], dtype=torch.float64) * n_samples
        count[:k] = float(n_samples)
        # drop non-finite steps
        nonfinite = ~torch.isfinite(total)
        total[nonfinite] = 0.0
        count[nonfinite] = 0.0

        for store, new in ((self._val_curves, total), (self._val_curve_counts, count)):
            old = store.get(name)
            store[name] = new if old is None or old.numel() != length else old + new

    def on_validation_epoch_end(self):
        '''
        Write the validation-averaged metric-vs-lead-time curves to log_dir as .npz.
        '''
        # all ranks must agree before the all_gather collective
        have_curves = torch.tensor(float(bool(self._val_curves)), device=self.device)
        if self.ddp:
            have_curves = self.all_gather(have_curves).min()
        if not float(have_curves):
            return

        names = sorted(self._val_curves)
        totals = torch.stack([self._val_curves[n] for n in names]).to(self.device)
        counts = torch.stack([self._val_curve_counts[n] for n in names]).to(self.device)
        if self.ddp:
            totals = self.all_gather(totals).sum(0)
            counts = self.all_gather(counts).sum(0)

        self._val_curves = {}
        self._val_curve_counts = {}

        if self.ddp and self.global_rank != 0:
            return

        curves = (totals / counts.clamp(min=1.0)).cpu().numpy()
        samples = counts.cpu().numpy()
        tag = f"_{self.val_tag}" if self.val_tag else ""
        os.makedirs(self.log_dir, exist_ok=True)
        out = f"{self.log_dir}val_curves{tag}_step{self.global_step}.npz"
        np.savez(out,
                 **{n: curves[i] for i, n in enumerate(names)},
                 **{f"n_samples_{n}": samples[i] for i, n in enumerate(names)})
        print(f"Wrote validation curves vs lead time ({len(names)} metrics) to {out}")

    @torch.no_grad()
    def predict_climate(self, 
            surface_feat_traj,
            multilevel_feat_traj,
            day_of_year_traj,
            hour_of_day_traj,
            constants_traj,
            yearly_constants_traj,
            return_pred=False, # for visualization
            z_pred = None,
            ensemble_size = 1,
            return_ens = False
            ):
        # surface_feat in shape [b, t, nlat, nlon, num_surface_feats]
        # multilevel_feat in shape [b, t, nlat, nlon, num_levels, num_multilevel_feats]
        # features are normalized

        surface_var_names = SURFACE_FEATURES 
        multilevel_var_names = MULTI_LEVEL_FEATURES
        
        surface_init = surface_feat_traj[:, 0] # b nlat nlon c
        multilevel_init = multilevel_feat_traj[:, 0] # b nlat nlon nlevel c
        scalar_params = assemble_scalar_params(day_of_year_traj, hour_of_day_traj, 0) # b, 2
        grid_params = assemble_grid_params(constants_traj, yearly_constants_traj, 0) # b nlat nlon (c + t*c)

        if z_pred is None:
            input_init = assemble_input(surface_init, multilevel_init) # b nlat nlon (c + nlevel*c)
            z_input = self.encode(input_init) # b zlat zlon z
            if ensemble_size > 1:
                z_input = z_input.repeat_interleave(ensemble_size, dim=0) # ens*b zlat zlon z
        else:
            z_input = z_pred # initialize rollout with prior latent

        surface_target = surface_feat_traj[:, 1:] # b t nlat nlon c
        multilevel_target = multilevel_feat_traj[:, 1:] # b t nlat nlon nlevel c

        surface_pred = torch.zeros_like(surface_target, device=surface_init.device) # b t nlat nlon c
        multilevel_pred = torch.zeros_like(multilevel_target, device=multilevel_init.device) # b t nlat nlon nlevel c

        if ensemble_size > 1:
            surface_pred = surface_pred.repeat_interleave(ensemble_size, dim=0) # ens*b t nlat nlon c
            multilevel_pred = multilevel_pred.repeat_interleave(ensemble_size, dim=0) # ens*b t nlat nlon nlevel c

        for t in range(surface_target.shape[1]):
            # assemble conditional info
            scalar_params = assemble_scalar_params(day_of_year_traj, hour_of_day_traj, t) # b, 2
            grid_params = assemble_grid_params(constants_traj, yearly_constants_traj, t) # b nlat nlon (c + t*c)

            if ensemble_size > 1:
                # repeat the grid and scalar params for ensemble size
                scalar_params = scalar_params.repeat_interleave(ensemble_size, dim=0) # ens*b 2
                grid_params = grid_params.repeat_interleave(ensemble_size, dim=0) # ens*b nlat nlon (c + t*c)

            # make prediction
            z_pred = self._timed_forward(z_input, scalar_params=scalar_params, grid_params=grid_params) # b zlat zlon z
            # decode the prediction
            model_pred = self.decode(z_pred) # b nlat nlon (c + nlevel*c)
            # rearrange prediction and save
            surface_pred_t, multilevel_pred_t = disassemble_input(model_pred, num_levels=multilevel_init.shape[-2], num_surface_channels=surface_init.shape[-1])
            surface_pred[:, t] = surface_pred_t
            multilevel_pred[:, t] =  multilevel_pred_t
            # update latent (latent unrolling)
            z_input = z_pred

        surface_pred, multilevel_pred = self.normalizer.batch_denormalize(surface_pred, multilevel_pred)
        surface_target, multilevel_target = self.normalizer.batch_denormalize(surface_target, multilevel_target)

        if ensemble_size > 1:
            if return_ens:
                surface_pred = rearrange(surface_pred, '(b ens) t nlat nlon c -> b ens t nlat nlon c', ens=ensemble_size) 
                multilevel_pred = rearrange(multilevel_pred, '(b ens) t nlat nlon nlevel c -> b ens t nlat nlon nlevel c', ens=ensemble_size) 
                
                multilevel_pred_flattened = rearrange(multilevel_pred, 'b ens t nlat nlon nlevel c -> b ens t nlat nlon (nlevel c)') # b ens t nlat nlon (nlevel c)
                multilevel_target_flattened = rearrange(multilevel_target, 'b t nlat nlon nlevel c -> b t nlat nlon (nlevel c)') # b t nlat nlon (nlevel c)
                
                pred_assembled = torch.cat([surface_pred, multilevel_pred_flattened], dim=-1) # b ens t nlat nlon (c + nlevel*c)
                target_assembled = torch.cat([surface_target, multilevel_target_flattened], dim=-1) # b t nlat nlon (c + nlevel*c)
                
                return None, pred_assembled, target_assembled, z_pred, None
            surface_pred = rearrange(surface_pred, '(b ens) t nlat nlon c -> b ens t nlat nlon c', ens=ensemble_size)
            multilevel_pred = rearrange(multilevel_pred, '(b ens) t nlat nlon nlevel c -> b ens t nlat nlon nlevel c', ens=ensemble_size)
        else:
            surface_pred = surface_pred.unsqueeze(1) # b 1 t nlat nlon c
            multilevel_pred = multilevel_pred.unsqueeze(1) # b 1 t nlat nlon nlevel c

        pred_ens_feat_dict = {}
        for c, surface_feat_name in enumerate(surface_var_names):
            pred_ens_feat_dict[surface_feat_name] = surface_pred[..., c] # b ens t nlat nlon
        for c, multilevel_feat_name in enumerate(multilevel_var_names):
            pred_ens_feat_dict[multilevel_feat_name] = multilevel_pred[..., c] # b ens t nlat nlon nlevel

        # the deterministic metrics are scored on the ensemble mean
        surface_pred = surface_pred.mean(1) # b t nlat nlon c
        multilevel_pred = multilevel_pred.mean(1) # b t nlat nlon nlevel c

        pred_feat_dict = {}
        target_feat_dict = {}
        for c, surface_feat_name in enumerate(surface_var_names):
            pred_feat_dict[surface_feat_name] = surface_pred[..., c]
            target_feat_dict[surface_feat_name] = surface_target[..., c]

        for c, multilevel_feat_name in enumerate(multilevel_var_names):
            pred_feat_dict[multilevel_feat_name] = multilevel_pred[..., c]
            target_feat_dict[multilevel_feat_name] = multilevel_target[..., c]

        loss_dict = {k:
                        latitude_weighted_rmse(pred_feat_dict[k], target_feat_dict[k],
                                                with_poles=self.config["data"]["with_poles"],
                                                nlon=self.config["data"]["nlon"],
                                                ) for k in pred_feat_dict.keys()} # b t for each key
        if not return_pred:
            return loss_dict
        else:
            return loss_dict, pred_feat_dict, target_feat_dict, z_pred, pred_ens_feat_dict
    
    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.lr, self.modelconfig,
                                    tag=f" {self.model_name}")
        if self.pde == "km_flow":
            step_size = 10
            gamma = 0.99
        else:
            step_size = 1
            gamma = 0.95
        scheduler = build_lr_scheduler(optimizer, self.config['training'], step_size, gamma)

        return [optimizer], [scheduler]
    