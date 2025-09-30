import torch
import torch.nn as nn

class ODEIntegrator:
    def __init__(self,
                 method='euler',  # 'euler' or 'heun' or 'midpoint'
                 ):
        self.method = method
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
            model(torch.cat((x, y), dim=-1), t.expand(x.shape[0]).unsqueeze(-1), **model_kwargs)

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
                 ):
        super(LinearScheduler, self).__init__()

        # for flow matching, the min_noise_std is not used
        self.num_train_timesteps = num_train_steps if num_train_steps is not None else num_refinement_steps + 1
        self.num_refinement_steps = num_refinement_steps
        self.sigmas = torch.linspace(0, 1,
                                     steps=self.num_train_timesteps)

        self.num_refinement_steps = num_refinement_steps
        self.ode_integrator = ODEIntegrator(method=integrator)

        self.training_criterion = nn.MSELoss()

        print(f"Using LinearScheduler with {self.num_train_timesteps} training steps and {self.num_refinement_steps} refinement steps.")

    def get_noise(self, size, device):
        return torch.randn(size, device=device)

    def compute_loss(self, x, y, model, eval=False, **kwargs):
        # x: [b nx ny d], conditioning. For PDEs this is u(t)
        # y: [b nx ny d], label. For PDEs this is u(t+dt)
        # cond: [b cond_dim]
        
        noise = self.get_noise(size=y.shape, device=y.device).to(y.dtype)

        # no need to train on k=0
        k = torch.randint(1, self.num_train_timesteps, device=x.device, size=(x.shape[0],)).long()

        # retrieve from the scheduler
        sigma_t = self.sigmas.to(x.device)[k] # noise coeff
        alpha_t = (1 - sigma_t) # signal coeff
        alpha_t = alpha_t.view(-1, *[1 for _ in range(y.ndim - 1)])
        sigma_t = sigma_t.view(-1, *[1 for _ in range(y.ndim - 1)])
        # Noise the label y
        y_noised = alpha_t * y + sigma_t * noise # y_t = alpha_t * y_0 + sigma_t * eps

        # conditional prediction. Concat condition (x) and noised input (y_noised)
        u_in = torch.cat([x, y_noised], dim=-1)  # input both condition and noised prediction, [b nx ny 2d]
        pred = model(u_in, k.float().view(-1, 1), **kwargs) # pred in shape [b nx ny d]
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

        timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, device=x.device).long()
        # trailing timesteps
        timesteps = timesteps[::((self.num_train_timesteps - 1) // refinement_steps)]
        sigmas = self.sigmas.to(x.device)[timesteps]

        # currently does not support noising input
        integrator = self.ode_integrator
        y_noised = integrator.integrate(x, y_noised, model, sigmas, timesteps, **kwargs)

        y = y_noised
        return y

    def forward(self, x, y, model, **kwargs):
        return self.compute_loss(x, y, model, **kwargs)