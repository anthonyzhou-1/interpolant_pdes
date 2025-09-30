import lightning as L
import torch
import pickle 
from einops import rearrange

from modules.models.FNO2D import FNO2d
from modules.models.FNO3D import FNO3d
from modules.models.DiT import DIT
from modules.models.ClimaDiT import ClimaDIT

from modules.diffusion.base import DiffusionScheduler
from modules.diffusion.flow_matching import LinearScheduler
from modules.diffusion.edm import EDMScheduler
from modules.diffusion.interpolant import DriftScheduler

from common.loss import ScaledLpLoss, PearsonCorrelationScore, latitude_weighted_rmse, sRMSE, VRMSE
from common.plotting import plot_result_2d, plot_entropy_2d
from common.climate_utils import plot_result_climate, plot_spectrum
from common.climate_utils import assemble_scalar_params, assemble_grid_params, assemble_input, disassemble_input
from dataset.plasim import SURFACE_FEATURES, MULTI_LEVEL_FEATURES

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
        self.eval_all = config['training'].get('eval_all', True) # flag to evaluate and plot all validation batches, not just the first one
        self.normalizer = normalizer
       
        self.criterion = ScaledLpLoss()
        self.correlation_criterion = PearsonCorrelationScore(reduce_batch=True)

        # flags for using probabilistic models or latent space models
        self.diffusion = False
        self.latent = self.modelconfig.get("latent", False)

        if self.model_name == "fno2d":
            fnoconfig = self.modelconfig["fno2d"]
            self.model = FNO2d(**fnoconfig)
            self.latent = False 

        elif self.model_name == "fno3d":
            fnoconfig = self.modelconfig["fno3d"]
            self.model = FNO3d(**fnoconfig)
            self.latent = False 

        elif self.model_name == "lns":
            ditconfig = self.modelconfig["lns"]
            if self.pde == "climate":
                self.model = ClimaDIT(**ditconfig)
            else:
                self.model = DIT(**ditconfig)
            self.latent = True
        
        elif self.model_name == "sfno":
            assert self.pde == "climate"
            try:
                from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperatorNet as SFNO
                from common.spherical_loss import L2LossS2
            except:
                raise ImportError
            
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

    def validation_step(self, batch, batch_idx, eval=False, z_pred=None, ensemble_size=1, return_ens=False):

        if self.pde == "climate":
            surface_feat, multi_level_feat, constants, yearly_constants, day_of_year, hour_of_day = batch    
        
            loss_dict, pred_feat_dict, target_feat_dict, z_pred = self.predict_climate(
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
            if batch_idx == 0:
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
                                    f'{self.log_dir}/val_t2m_{self.current_epoch}.png')
                    plot_result_climate(z500_pred,
                                    z500_target,
                                    f'{self.log_dir}/val_z500_{self.current_epoch}.png')
                    plot_result_climate(pr_6h_pred,
                                    pr_6h_target,
                                    f'{self.log_dir}/val_pr_6h_{self.current_epoch}.png')
                    plot_result_climate(u250_pred,
                                    u250_target,
                                    f'{self.log_dir}/val_u250_{self.current_epoch}.png')
                    plot_result_climate(t850_pred,
                                    t850_target,
                                    f'{self.log_dir}/val_t850_{self.current_epoch}.png')
                    
                    plot_spectrum(t2m_pred,
                                    t2m_target,
                                    f'{self.log_dir}/val_t2m_spectrum_{self.current_epoch}.png')
                    plot_spectrum(z500_pred,
                                    z500_target,
                                    f'{self.log_dir}/val_z500_spectrum_{self.current_epoch}.png')
                    plot_spectrum(pr_6h_pred,
                                    pr_6h_target,
                                    f'{self.log_dir}/val_pr_6h_spectrum_{self.current_epoch}.png')
                    plot_spectrum(u250_pred,
                                    u250_target,
                                    f'{self.log_dir}/val_u250_spectrum_{self.current_epoch}.png')
                    plot_spectrum(t850_pred,
                                    t850_target,
                                    f'{self.log_dir}/val_t850_spectrum_{self.current_epoch}.png')
            
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
            at_correlation = False

            if self.latent:
                u_input = self.encode(u_input, cond) # encode input to latent space

            for i in range(0, nt-1):
                pred = self.forward(u_input, cond) # shape (b, nx, ny, 1)
                u_true = u_label[:, i+1]

                if pred.isnan().any():
                    print(f"NaN detected in prediction at step {i+1}.")
                    break

                true_denorm = self.normalizer.denormalize(u_true)
                if self.latent:
                    pred_u = self.decode(pred, cond) # decode prediction from latent space
                else:
                    pred_u = pred

                pred_denorm = self.normalizer.denormalize(pred_u)

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

                u_input = pred # update input for next step
            
            if not at_correlation:
                correlation_time = nt-1 # didn't go below correlation threshold, therefore the time is the last step

            loss = sum(accumulated_loss) / len(accumulated_loss)
            VRMSE_total = sum(accumulated_VRMSE) / len(accumulated_VRMSE)

            len_rollout = len(accumulated_VRMSE)
            for i in range(10):
                start = i * (len_rollout // 10)
                end = (i + 1) * (len_rollout // 10)
                rollout_loss_i = sum(accumulated_VRMSE[start:end]) / 10 
                self.log(f'val/VRMSE_{i}', rollout_loss_i, on_step=False, on_epoch=True, sync_dist=self.ddp)

            self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/correlation_time', correlation_time, on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/VRMSE', VRMSE_total, on_step=False, on_epoch=True, sync_dist=self.ddp)

            self.log('val/sRMSE_low', sum(sRMSE_low) / len(sRMSE_low), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/sRMSE_mid', sum(sRMSE_mid) / len(sRMSE_mid), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/sRMSE_high', sum(sRMSE_high) / len(sRMSE_high), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/sRMSE_total', sum(sRMSE_total) / len(sRMSE_total), on_step=False, on_epoch=True, sync_dist=self.ddp)

            if batch_idx == 0 or self.eval_all: # only plot first batch or if eval_all flag is set
                if self.ddp and self.global_rank != 0:
                    pass
                else:
                    u_label_denorm = self.normalizer.denormalize(u_label)

                    if len(u_label_denorm.shape) > 5: # b nt nx ny nz c
                        u_label_denorm = u_label_denorm[:, :, 0] 
                        u_pred_denorm = u_pred_denorm[:, :, 0] 

                    plot_result_2d(u_label_denorm, u_pred_denorm, n_t=10, path=f'{self.log_dir}batch_{batch_idx}_ep_{self.current_epoch}.png')
                    if self.pde == "km_flow":
                        plot_entropy_2d(u_label_denorm, u_pred_denorm, n_t=5, path=f'{self.log_dir}batch_{batch_idx}_entropy_ep_{self.current_epoch}.png')
                    
                    with open(f'{self.log_dir}accumulated_loss_batch_{batch_idx}_ep{self.current_epoch}.pkl', 'wb') as f:
                        pickle.dump(accumulated_loss, f)

                    with open(f'{self.log_dir}accumulated_VRMSE_batch_{batch_idx}_ep{self.current_epoch}.pkl', 'wb') as f:
                        pickle.dump(accumulated_VRMSE, f)

                    with open(f'{self.log_dir}sRMSE_low_batch_{batch_idx}_ep{self.current_epoch}.pkl', 'wb') as f:
                        pickle.dump(sRMSE_low, f)

                    with open(f'{self.log_dir}sRMSE_mid_batch_{batch_idx}_ep{self.current_epoch}.pkl', 'wb') as f:
                        pickle.dump(sRMSE_mid, f)

                    with open(f'{self.log_dir}sRMSE_high_batch_{batch_idx}_ep{self.current_epoch}.pkl', 'wb') as f:
                        pickle.dump(sRMSE_high, f)
    
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
        b = surface_target.shape[0]

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
            z_pred = self.forward(z_input, scalar_params=scalar_params, grid_params=grid_params) # b zlat zlon z
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
                
                return None, pred_assembled, target_assembled, z_pred
            if b == 1:
                surface_pred = surface_pred.mean(dim=0, keepdim=True) # ens t nlat nlon c -> 1 t nlat nlon c
                multilevel_pred = multilevel_pred.mean(dim=0, keepdim=True) # ens t nlat nlon nlevel c -> 1 t nlat nlon nlevel c
            else:
                surface_pred = rearrange(surface_pred, '(b ens) t nlat nlon c -> b ens t nlat nlon c', ens=ensemble_size).mean(1) # b t nlat nlon c
                multilevel_pred = rearrange(multilevel_pred, '(b ens) t nlat nlon nlevel c -> b ens t nlat nlon nlevel c', ens=ensemble_size).mean(1) # b t nlat nlon nlevel c

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
            return loss_dict, pred_feat_dict, target_feat_dict, z_pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.pde == "km_flow":
            step_size = 10
            gamma = 0.99
        else:
            step_size = 1
            gamma = 0.95
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        return [optimizer], [scheduler]
    