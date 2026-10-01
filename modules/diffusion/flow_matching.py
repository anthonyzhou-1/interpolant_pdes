import torch
import torch.nn as nn

class ODEIntegrator:
    def __init__(self,
                 method='euler',  # 'euler' or 'heun' or 'midpoint'
                 t_scale=1.0,
                 ):
        self.method = method
        self.t_scale = t_scale
        assert method in ['euler', 'heun', 'midpoint'], 'Method not implemented'

    def step_fn(self, x, fn, dt, ts, model_kwargs):
        method = self.method
        if ts[1] == 0 and self.method == 'heun': # prevent irregularity at last time
            method = 'euler'
        if method == 'euler':
            return x + dt * fn(x, ts[0], model_kwargs)
        elif method == 'heun':
            dx = fn(x, ts[0], model_kwargs)
            x1 = x + dt * dx
            return x + 0.5 * dt * (dx + fn(x1, ts[1], model_kwargs))
        elif method == 'midpoint':
            x1 = x + 0.5 * dt * fn(x, ts[0], model_kwargs)
            return x + dt * fn(x1, ts[1], model_kwargs)

    def integrate(self, x, y, model,
                  stencils, timesteps,
                  **kwargs):
        model_wrapper_fn = lambda y, t, model_kwargs: \
            model(torch.cat((x, y), dim=-1), self.t_scale * t.expand(x.shape[0]).unsqueeze(-1), **model_kwargs)

        for i_t in range(len(stencils)-1):
            t_current = stencils[i_t] # sigma_t
            t_next = stencils[i_t+1] # sigma_t+1
            dt = t_next - t_current # (sigma_t+1 - sigma_t)
            if self.method != 'midpoint':
                y = self.step_fn(y, model_wrapper_fn, dt,
                                 [timesteps[i_t], timesteps[i_t+1]],
                                 kwargs)
            else:
                y = self.step_fn(y, model_wrapper_fn, dt,
                                 [timesteps[i_t], (timesteps[i_t+1] + timesteps[i_t]) / 2],
                                 kwargs)
        return y

class LinearScheduler(nn.Module):
    def __init__(self,
                 num_refinement_steps,  # this corresponds to physical time steps
                 num_train_steps=None,  # number of training steps
                 integrator='euler',  # 'euler' or 'heun' or 'midpoint', worth noting that this only available for flow
                 continuous_t=True,
                 t_scale=1000.0,
                 ):
        super(LinearScheduler, self).__init__()

        self.num_train_timesteps = num_train_steps if num_train_steps is not None else num_refinement_steps + 1
        self.num_refinement_steps = num_refinement_steps
        # continuous_t=False conditions on a discrete grid index instead of t
        self.continuous_t = continuous_t
        # scale t in [0, 1] up for the sinusoidal timestep embedding
        self.t_scale = t_scale if continuous_t else 1.0
        self.sigmas = torch.linspace(0, 1,
                                     steps=self.num_train_timesteps)

        self.ode_integrator = ODEIntegrator(method=integrator, t_scale=self.t_scale)

        self.training_criterion = nn.MSELoss()

        print(f"Using LinearScheduler with {self.num_train_timesteps} training steps and {self.num_refinement_steps} refinement steps.")
        print(f"continuous_t: {self.continuous_t}, t_scale: {self.t_scale}")

    def get_noise(self, size, device):
        return torch.randn(size, device=device)

    def compute_loss(self, x, y, model, eval=False, **kwargs):
        # x: [b nx ny d], conditioning. For PDEs this is u(t)
        # y: [b nx ny d], label. For PDEs this is u(t+dt)
        # cond: [b cond_dim]
        
        noise = self.get_noise(size=y.shape, device=y.device).to(y.dtype)

        # t ~ U(0, 1]; t=0 is the noiseless end of the path
        if self.continuous_t:
            t = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) * (1.0 - 1e-5) + 1e-5
        else:
            k = torch.randint(1, self.num_train_timesteps, device=x.device, size=(x.shape[0],)).long()
            t = self.sigmas.to(x.device)[k]

        sigma_t = t # noise coeff
        alpha_t = (1 - sigma_t) # signal coeff
        alpha_t = alpha_t.view(-1, *[1 for _ in range(y.ndim - 1)])
        sigma_t = sigma_t.view(-1, *[1 for _ in range(y.ndim - 1)])
        # Noise the label y
        y_noised = alpha_t * y + sigma_t * noise # y_t = alpha_t * y_0 + sigma_t * eps

        # conditional prediction. Concat condition (x) and noised input (y_noised)
        u_in = torch.cat([x, y_noised], dim=-1)  # input both condition and noised prediction, [b nx ny 2d]
        cond_t = self.t_scale * t.float().view(-1, 1) if self.continuous_t else k.float().view(-1, 1)
        pred = model(u_in, cond_t, **kwargs) # pred in shape [b nx ny d]
        target = noise - y # predict eps - y
        loss = self.training_criterion(pred, target)
        if eval:
            return loss, pred, target
        return loss

    def sample(self, x, model, refinement_steps=None, **kwargs):
        
        if refinement_steps is None:
            refinement_steps = self.num_refinement_steps

        # x: [b nlat nlon d]
        y_noised = self.get_noise(
            x.shape, device=x.device
        ).to(x.dtype)

        if self.continuous_t:
            # sigma_t == t, descending from 1 to 0
            timesteps = torch.linspace(1, 0, refinement_steps + 1, device=x.device)
            sigmas = timesteps
        else:
            timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, device=x.device).long()
            # trailing timesteps
            timesteps = timesteps[::((self.num_train_timesteps - 1) // refinement_steps)]
            sigmas = self.sigmas.to(x.device)[timesteps]

        integrator = self.ode_integrator
        y_noised = integrator.integrate(x, y_noised, model, sigmas, timesteps, **kwargs)

        y = y_noised
        return y

    def forward(self, x, y, model, **kwargs):
        return self.compute_loss(x, y, model, **kwargs)