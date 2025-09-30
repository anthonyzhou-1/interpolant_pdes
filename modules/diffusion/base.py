import torch
import numpy as np
import torch.nn as nn

class DiffusionScheduler(nn.Module):
    def __init__(self, 
                 schedule="linear",
                 noise_steps=100,
                 num_ddim_steps=10,
                 skip_percent=0.0,
                 beta_start=0.0001,
                 beta_end=0.02,
                 scale=400,
                 mode="ddpm",
                 ndim=2):
        super(DiffusionScheduler, self).__init__()

        self.noise_steps = noise_steps
        self.num_ddim_steps = num_ddim_steps
        self.skip_percent = skip_percent
        self.scale = scale
        self.mode = mode # ddpm, ddim, tsm

        self.schedule = schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.ndim = ndim

        self.criterion = nn.MSELoss()

        clip_min = 1e-9

        if self.schedule == 'cosine':
            s = 0.008
            tlist = torch.arange(1, self.noise_steps+1, 1)
            temp1 = torch.cos((tlist/self.noise_steps+s)/(1+s)*np.pi/2)
            temp1 = temp1*temp1
            temp2 = np.cos(((tlist-1)/self.noise_steps+s)/(1+s)*np.pi/2)
            temp2 = temp2*temp2

            self._beta_source = 1-(temp1/temp2)
            self._beta_source[self._beta_source > 0.999] = 0.999
            self._beta = torch.cat((torch.tensor([0]), self._beta_source), dim=0)[1:]
            self._alpha = 1-self._beta
            self._alpha_hat = torch.cumprod(self._alpha, 0)
        elif self.schedule == 'simple_linear':
            t = torch.linspace(0, 1, self.noise_steps)
            self._alpha_hat = torch.clip(1-t, min=clip_min, max=1)
            self._alpha = torch.div(self._alpha_hat, torch.cat((torch.tensor([1]), self._alpha_hat[:-1]), 0))
            self._beta = 1 - self._alpha
        else:
            self._beta = self.prepare_noise_schedule()
            self._alpha = 1. - self._beta
            self._alpha_hat = torch.cumprod(self._alpha, dim=0)
            self._alpha_hat = torch.clip(self._alpha_hat, min=clip_min, max=1)
        
        self.register_buffer('beta', self._beta)
        self.register_buffer('alpha', self._alpha)
        self.register_buffer('alpha_hat', self._alpha_hat)

        print(f"Mode: {self.mode}")

    def prepare_noise_schedule(self):
        """
        Creates the noise schedule beta
        """
        if self.schedule == 'linear':
            scale = (self.scale/self.noise_steps)
            self.beta_start = self.beta_start * scale
            self.beta_end = self.beta_end * scale
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.schedule == 'cosine':
            timesteps = torch.linspace(0, 1, self.noise_steps)
            betas = self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (1 + np.cos(np.pi/2 * timesteps)**2)
            return torch.tensor(betas, dtype=torch.float)
        else:
            return NotImplementedError("Noising Schedule Not Implemented!")
        
    def sample_timesteps(self, n, device='cpu'):
        """
        Returns random outputs for diff_time for training only 
        """
        low = int(self.skip_percent * self.noise_steps)
        return torch.randint(low=low, high=self.noise_steps, size=(n,), device=device)
    
    def wide(self, t):
        if self.ndim == 2:
            return t[:, None, None, None]
        elif self.ndim == 3:
            return t[:, None, None, None, None]
    
    def noise_states(self, x, t):
        """
        Noises a batch of states
        """
        sqrt_alpha_hat = self.wide(torch.sqrt(self.alpha_hat[t]))
        sqrt_one_minus_alpha_hat = self.wide(torch.sqrt(1 - self.alpha_hat[t]))

        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps
    
    def compute_loss(self, x, y, model, **kwargs):
        t = self.sample_timesteps(x.shape[0], device=x.device)
        skip_t = int(self.skip_percent * self.noise_steps)

        # don't train on skipped steps
        if self.skip_percent > 0:
            indices = torch.nonzero(torch.lt(t, skip_t)).squeeze(-1)                
            t[indices] = skip_t

        y_noised, noise = self.noise_states(y, t)
        predicted_noise = model(torch.cat([x, y_noised], dim=-1), t, **kwargs) # 1
        loss = self.criterion(noise, predicted_noise)      
        return loss 
    
    def setup_ddim_sampling(self):
        noise_steps = self.noise_steps

        delta = noise_steps//self.num_ddim_steps
        ref_arr = list(reversed(range(0, noise_steps, delta)))
        ref_arr = np.array(ref_arr)/noise_steps

        for val in ref_arr:
            assert val < 1, val
            assert val >= 0, val            

        self.ref_arr = ref_arr
    
    def sample_ddim(self, initial_cond, model, **kwargs):

        ref_arr = self.ref_arr

        n = initial_cond.shape[0]
        device = initial_cond.device

        with torch.no_grad():
            x = torch.randn_like(initial_cond, device=device)

            denoising_steps = self.noise_steps

            for idx, i in enumerate(ref_arr):
                t = (torch.ones(n, device=device) * int(i * denoising_steps)).long()
                
                if idx > 0:
                    sqrt_alpha_hat = self.wide(torch.sqrt(self.alpha_hat[t]))
                    sqrt_one_minus_alpha_hat = self.wide(torch.sqrt(1 - self.alpha_hat[t]))

                    x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * predicted_noise
                                        
                predicted_noise = model(torch.cat([initial_cond, x], dim=-1), t, **kwargs)               
                
                sqrt_alpha_hat = self.wide(torch.sqrt(self.alpha_hat[t]))
                sqrt_one_minus_alpha_hat = self.wide(torch.sqrt(1 - self.alpha_hat[t]))

                x = (x - sqrt_one_minus_alpha_hat * predicted_noise)/sqrt_alpha_hat

        return x
    
    def sample(self, initial_cond, model, **kwargs):
        if self.mode == "ddpm" or self.mode == "tsm":
            return self.sample_main(initial_cond, model, **kwargs)
        elif self.mode == "ddim":
            if not hasattr(self, 'ref_arr'):
                self.setup_ddim_sampling()
            return self.sample_ddim(initial_cond, model, **kwargs)
        else:
            raise NotImplementedError(f"Sampling method {self.mode} not implemented!")

    def sample_main(self, initial_cond, model, **kwargs):
        """
        Main sample loop for DDPMs & TSMs
        """

        n = initial_cond.shape[0]
        device = initial_cond.device
        
        with torch.no_grad():
            x = torch.randn_like(initial_cond, device=device)

            denoising_steps = self.noise_steps

            for i in reversed(range(0, denoising_steps)):
                t = (torch.ones(n, device=device) * i).long()
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                predicted_noise = model(torch.cat([initial_cond, x], dim=-1), t, **kwargs)

                if i == int(self.skip_percent * denoising_steps) and self.skip_percent > 0:
                    sqrt_alpha_hat = self.wide(torch.sqrt(self.alpha_hat[t]))
                    sqrt_one_minus_alpha_hat = self.wide(torch.sqrt(1 - self.alpha_hat[t]))

                    x = (x - sqrt_one_minus_alpha_hat * predicted_noise)/sqrt_alpha_hat
                    break

                alpha = self.wide(self.alpha[t])
                alpha_hat = self.wide(self.alpha_hat[t])
                beta = self.wide(self.beta[t])

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                                
        return x