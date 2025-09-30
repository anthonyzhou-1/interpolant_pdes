import torch
import lightning as L
from common.loss import ScaledLpLoss, KL_Loss, WeightedLoss, latitude_weighted_rmse, VRMSE, sRMSE
from common.plotting import plot_result_2d
from modules.layers.distributions import DiagonalGaussianDistribution
from common.climate_utils import assemble_input, disassemble_input, plot_spectrum, plot_result_climate
from dataset.plasim import SURFACE_FEATURES, MULTI_LEVEL_FEATURES
import pickle 

class AutoencoderModule(L.LightningModule):
    def __init__(self,
                 config,
                 normalizer=None
                 ):
        super().__init__()

        self.config=config
        if config['training']['strategy'] == 'ddp' or config['training']['strategy'] == 'ddp_find_unused_parameters_true':
            self.ddp = True
        else:
            self.ddp = False
            
        self.normalizer = normalizer
        self.lr = config['model']['lr']
        self.log_dir = config['training']['log_dir']
        self.pde = config['data']['pde']

        if self.pde == "climate":
            from modules.models.KL_AE import Encoder, Decoder

            self.encoder = Encoder(in_channels=config["model"]["surface_dim"] + config["model"]["multilevel_dim"],
                                    hidden_channels=config["model"]["channel_dim"],
                                    z_channels=config["model"]["latent_dim"] * 2)
            self.decoder = Decoder(out_channels=config["model"]["surface_dim"] + config["model"]["multilevel_dim"],
                                   hidden_channels=config["model"]["channel_dim"],
                                   z_channels= config["model"]["latent_dim"])
            self.criterion = WeightedLoss(loss_fn=torch.nn.MSELoss(reduction='none'),
                                     latitude_resolution=config["data"]["nlat"],
                                     longitude_resolution=config["data"]["nlon"],
                                     with_poles=False)
            self.loss = KL_Loss(kl_weight=config["model"]["kl_weight"],
                                     criterion=self.criterion)
            self.saturate_z = False
            self.kl = True
        elif self.pde == "MHD_64":
            from modules.models.KL_AE import Encoder, Decoder

            self.encoder = Encoder(**config['model']['encoder'])
            self.decoder = Decoder(**config['model']['decoder'])
            self.criterion = ScaledLpLoss()
            self.saturate_z = True
            self.kl = False
        else:
            from modules.models.DC_AE import Encoder, Decoder
            encoderconfig = config['model']['encoder']
            decoderconfig = config['model']['decoder']

            self.encoder = Encoder(**encoderconfig)
            self.decoder = Decoder(**decoderconfig)
            self.criterion = ScaledLpLoss()
            self.saturate_z = True
            self.kl = False

        self.save_hyperparameters()

    def saturate(self, x, B=5.0):
        x = x /torch.sqrt(1 + x**2/B**2)
        return x

    def encode(self, x, cond=None):
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)  # b nx ny c -> b c nx ny
        elif len(x.shape) == 5: # b nx ny nz c -> b c nx ny nz
            x = x.permute(0, 4, 1, 2, 3) 
        z = self.encoder(x, cond)
        posterior = None

        if self.saturate_z:
            z = self.saturate(z)

        if self.kl:
            posterior = DiagonalGaussianDistribution(z)
            z = posterior.sample()
            
        return z, posterior

    def decode(self, z, cond=None):
        dec = self.decoder(z, cond)
        if len(dec.shape) == 4:
            dec = dec.permute(0, 2, 3, 1) # b c nx ny -> b nx ny c
        elif len(dec.shape) == 5: # b c nx ny nz -> b nx ny nz c
            dec = dec.permute(0, 2, 3, 4, 1)
        return dec

    def forward(self, x, cond=None):
        z, posterior = self.encode(x, cond)
        dec = self.decode(z, cond)
        return dec, posterior
    
    def get_inputs_climate(self, batch):
        surface_feat, multi_level_feat, constants, yearly_constants, day_of_year, hour_of_day = batch  
        inputs = assemble_input(surface_feat, multi_level_feat) # b nlat nlon (c + nlevel*c)
        cond = None 
        return inputs, cond
    
    def get_inputs(self, batch):
        inputs = batch["input_fields"] # b nx ny c
        if len(inputs.shape) > 4: # b 1 nx ny c
            inputs = inputs[:, 0] # b nx ny c
        cond = batch.get("constant_scalars", None) # b num_cond
        if self.pde == "rayleigh_benard":
            cond[:, 0] = torch.log10(cond[:, 0]) # log Rayleigh number
        
        return inputs, cond

    def training_step(self, batch, batch_idx):
        if self.pde == "climate":
            inputs, cond = self.get_inputs_climate(batch)
            reconstructions, posterior = self(inputs, cond)
            loss, log_dict = self.loss(inputs, reconstructions, posterior, split="train")
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=self.ddp)
        else:
            inputs, cond = self.get_inputs(batch)
            reconstructions, posterior = self(inputs, cond)
            loss = self.criterion(reconstructions, inputs)
            self.log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
        return loss 

    def validation_step(self, batch, batch_idx):
        if self.pde == "climate":
            surface_var_names = SURFACE_FEATURES 
            multilevel_var_names = MULTI_LEVEL_FEATURES
            surface_feat, multi_level_feat, _, _, _, _ = batch  
            inputs, cond = self.get_inputs_climate(batch)
            reconstructions, posterior = self(inputs, cond)

            surface_reconstruction, multilevel_reconstruction = disassemble_input(reconstructions)

            surface_feat, multi_level_feat = self.normalizer.batch_denormalize(surface_feat, multi_level_feat)
            surface_reconstruction, multilevel_reconstruction = self.normalizer.batch_denormalize(surface_reconstruction, multilevel_reconstruction)

            loss, log_dict = self.loss(assemble_input(surface_feat, multi_level_feat), 
                                    assemble_input(surface_reconstruction, multilevel_reconstruction),
                                    posterior,
                                    split="val")

            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=self.ddp)

            pred_feat_dict = {}
            target_feat_dict = {}
            for c, surface_feat_name in enumerate(surface_var_names):
                pred_feat_dict[surface_feat_name] = surface_reconstruction[..., c] # b nlat nlon
                target_feat_dict[surface_feat_name] = surface_feat[..., c] # b nlat nlon

            for c, multilevel_feat_name in enumerate(multilevel_var_names):
                pred_feat_dict[multilevel_feat_name] = multilevel_reconstruction[..., c] # b nlat nlon nlevel 
                target_feat_dict[multilevel_feat_name] = multi_level_feat[..., c] # b nlat nlon nlevel

            loss_dict = {k:
                    latitude_weighted_rmse(pred_feat_dict[k], target_feat_dict[k],
                                            with_poles=False,
                                            nlon=self.config["data"]["nlon"],
                                            nlat=self.config["data"]["nlat"],
                                            with_time=False,
                                            ) for k in pred_feat_dict.keys()} # b nlat nlon or b nlat nlon nlevel
            

            if batch_idx == 0:
                if self.ddp and self.global_rank != 0:
                    pass
                else: # not ddp or rank 0
                    t2m_pred = pred_feat_dict['tas'][0].unsqueeze(0).cpu().numpy()
                    t2m_target = target_feat_dict['tas'][0].unsqueeze(0).cpu().numpy()
                    z500_pred = pred_feat_dict['zg'][0, ..., 7].unsqueeze(0).cpu().numpy()
                    z500_target = target_feat_dict['zg'][0, ..., 7].unsqueeze(0).cpu().numpy()
                    pr_6h_pred = pred_feat_dict['pr_6h'][0].unsqueeze(0).cpu().numpy()
                    pr_6h_target = target_feat_dict['pr_6h'][0].unsqueeze(0).cpu().numpy()
                    u250_pred = pred_feat_dict['ua'][0, ..., 4].unsqueeze(0).cpu().numpy()
                    u250_target = target_feat_dict['ua'][0, ..., 4].unsqueeze(0).cpu().numpy()
                    t850_pred = pred_feat_dict['ta'][0, ..., 10].unsqueeze(0).cpu().numpy()
                    t850_target = target_feat_dict['ta'][0, ..., 10].unsqueeze(0).cpu().numpy()

                    plot_result_climate(t2m_pred, # t h w
                                    t2m_target,
                                    f'{self.log_dir}/val_t2m_{self.current_epoch}.png',
                                    num_t=1)
                    plot_result_climate(z500_pred,
                                    z500_target,
                                    f'{self.log_dir}/val_z500_{self.current_epoch}.png',
                                    num_t=1)
                    plot_result_climate(pr_6h_pred,
                                    pr_6h_target,
                                    f'{self.log_dir}/val_pr_6h_{self.current_epoch}.png',
                                    num_t=1)
                    plot_result_climate(u250_pred,
                                    u250_target,
                                    f'{self.log_dir}/val_u250_{self.current_epoch}.png',
                                    num_t=1)
                    plot_result_climate(t850_pred,
                                    t850_target,
                                    f'{self.log_dir}/val_t850_{self.current_epoch}.png',
                                    num_t=1)
                    
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

            # calculate the mean loss, shape b for each key, b l for multilevel keys
            t2m_loss = loss_dict['tas'].mean() # surface temp, mean across batch dim
            pr_6h_loss = loss_dict['pr_6h'].mean() # 6-hour accumulated precipitation
            z500_loss = loss_dict['zg'][..., 7].mean() # geopotential at level=7
            u250_loss = loss_dict['ua'][..., 4].mean() # u wind at level=4
            t850_loss = loss_dict['ta'][..., 10].mean() # temp at level=10
            
            self.log('val/t2m', t2m_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp) 
            self.log('val/pr_6h', pr_6h_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/z500', z500_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/u250', u250_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/t850', t850_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        else:
            inputs, cond = self.get_inputs(batch)

            reconstructions, posterior = self(inputs, cond) # b nx ny c

            inputs_denorm = self.normalizer.denormalize(inputs)
            reconstructions_denorm = self.normalizer.denormalize(reconstructions) 

            loss = self.criterion(reconstructions_denorm, inputs_denorm)

            vrmse = VRMSE(reconstructions_denorm, inputs_denorm)

            if len(inputs.shape) == 4:
                spatial = 2
            elif len(inputs.shape) == 5:
                spatial = 3

            sRMSE_all = sRMSE(reconstructions_denorm, inputs_denorm, spatial=spatial) # calculate sRMSE

            if batch_idx == 0:
                if self.ddp and self.global_rank != 0:
                    pass
                else:
                    if len(inputs.shape) == 5: # b nx ny nz c
                        inputs_denorm = inputs_denorm[:, :, 0]
                        reconstructions_denorm = reconstructions_denorm[:, :, 0]
                    plot_result_2d(inputs_denorm.unsqueeze(1), # b t x y 1
                                reconstructions_denorm.unsqueeze(1), 
                                n_t=1, 
                                path=f'{self.log_dir}ep_{self.current_epoch}.png')

            self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/vrmse', vrmse, on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/sRMSE_low', sRMSE_all[0], on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/sRMSE_mid', sRMSE_all[1], on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log('val/sRMSE_high', sRMSE_all[2], on_step=False, on_epoch=True, sync_dist=self.ddp)
        
        return loss 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters()), lr=self.lr)
        if self.pde == "km_flow":
            step_size = 10
        else:
            step_size = 1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.99)

        return [optimizer], [scheduler]