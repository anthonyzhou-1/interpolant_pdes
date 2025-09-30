import torch
import torch.nn as nn

class Integrator:
    def __init__(self,
                 method='em', 
                 ):
        self.method = method

    def step_fn(self, x, drift, dt, g):
        if self.method == 'em': # Euler-Maruyama
            dW = torch.sqrt(dt)*torch.randn_like(x, device=x.device)
            return x + dt*drift + g*dW
        elif self.method == "euler": # ODE
            return x + dt*drift 

    def integrate(self, x, y, model, timesteps, g_fn, use_gF=False, scheduler=None, **kwargs):

        for i_t in range(len(timesteps)-1):
            t_current = timesteps[i_t]
            t_next = timesteps[i_t+1] 
            dt = t_next - t_current 
            g_t = g_fn(t_current.expand(x.shape[0]))  # shape (b, 1, 1, 1)

            drift = model(torch.cat((x, y), dim=-1), t_current.expand(x.shape[0]).unsqueeze(-1), **kwargs)

            # if g is equal to sigma then the score vanishes during sampling, otherwise need to compute
            if use_gF:
                t = t_current.expand(x.shape[0])
                alpha_t = scheduler.alpha(t)
                beta_t = scheduler.beta(t)
                alpha_dot_t = scheduler.alpha_dot(t)
                beta_dot_t = scheduler.beta_dot(t)
                sigma_t = scheduler.sigma(t)
                sigma_dot_t = scheduler.sigma_dot(t)
                denom = scheduler.wide(t) * sigma_t * (beta_dot_t * sigma_t - beta_t * sigma_dot_t)
                numerator = beta_dot_t*y + (beta_t*alpha_dot_t - beta_dot_t*alpha_t)*x
                score = (beta_t * drift - numerator) / denom
                drift = drift + 0.5*(g_t**2 - sigma_t**2) * score
            
            y = self.step_fn(y, drift, dt, g_t)
        return y

class DriftScheduler(nn.Module):
    def __init__(self,
                 num_refinement_steps,  # this corresponds to physical time steps
                 num_train_steps=None,  # number of training steps
                 integrator='em', 
                 sigma_coef=1.0,  
                 beta_fn = "t",
                 use_gF = False,
                 antithetic_sampling=True,
                 sigma_sample=None,
                 ndim=2,
                 ):
        super(DriftScheduler, self).__init__()

        self.num_train_timesteps = num_train_steps if num_train_steps is not None else num_refinement_steps + 1
        self.num_refinement_steps = num_refinement_steps
        self.sigma_coef = sigma_coef
        self.method = integrator
        self.integrator = Integrator(method=integrator)

        self.ndim = ndim
        self.beta_fn = beta_fn
        self.use_gF = use_gF
        self.antithetic_sampling = antithetic_sampling
        self.sigma_sample = sigma_sample if sigma_sample is not None else sigma_coef

        print(f'Scheduler initialized with {self.num_train_timesteps} training steps and {self.num_refinement_steps} refinement steps.')
        print(f"sigma_coef: {self.sigma_coef}, integrator: {integrator}, beta_fn: {self.beta_fn}, use_gf: {self.use_gF}, antithetic_sampling: {self.antithetic_sampling}")

    def wide(self, t):
        if self.ndim == 2:
            return t[:, None, None, None]
        elif self.ndim == 3:
            return t[:, None, None, None, None]

    def alpha(self, t):
        return self.wide(1-t) 

    def alpha_dot(self, t):
        return self.wide(-1.0 * torch.ones_like(t))

    def beta(self, t):  
        if self.beta_fn == "t":
            return self.wide(t)
        elif self.beta_fn == "t^2":
            return self.wide(t**2)

    def beta_dot(self, t):
        if self.beta_fn == "t":
            return self.wide(torch.ones_like(t))
        elif self.beta_fn == "t^2":
            return self.wide(2.0 * t)
    
    def sigma(self, t, sample=False):
        if sample:
            return self.sigma_sample * self.wide(1-t)
        else:
            return self.sigma_coef * self.wide(1-t)

    def sigma_dot(self, t, sample=False):
        if sample:
            return self.sigma_sample * self.wide(-1.0 * torch.ones_like(t))
        else:
            return self.sigma_coef * self.wide(-1.0 * torch.ones_like(t))
    
    # derived for beta(t) = t, so can't use if beta_fn is not "t"
    def g_F(self, t):
        return self.sigma_coef * self.wide(torch.sqrt((1-t)**2 + 2*(1-t)))
    
    def I(self, x0, x1, t):
        return self.alpha(t) * x0 + self.beta(t) * x1
    
    def dIdt(self, x0, x1, t):
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

    def get_noise(self, size, device):
        return torch.randn(size, device=device)
    
    def image_sq_norm(self, x):
        return x.pow(2).sum(-1).sum(-1).sum(-1)

    def compute_loss(self, x, y, model, **kwargs):
        # x: [b nx ny d], source distribution. For PDEs this is u(t)
        # y: [b nx ny d], target distribution. For PDEs this is u(t+dt)
        # cond: [b cond_dim]
        
        noise = self.get_noise(size=y.shape, device=y.device).to(y.dtype)

        # no need to train on t=1
        t = torch.randint(0, self.num_train_timesteps-1, device=x.device, size=(x.shape[0],)) / (self.num_train_timesteps - 1)  # shape (b,)

        dIdt = self.dIdt(x, y, t) # shape (b, nx, ny, d)
        I = self.I(x, y, t) # shape (b, nx, ny, d)

        sigma_dot = self.sigma_dot(t) # shape (b, 1, 1, 1) 
        sigma = self.sigma(t) # shape (b, 1, 1, 1)

        W = self.wide(torch.sqrt(t)) * noise

        if self.antithetic_sampling:
            I_p = I + sigma * W
            I_m = I - sigma * W
            model_in_p = torch.cat([x, I_p], dim=-1)
            model_in_m = torch.cat([x, I_m], dim=-1)
            target_p = dIdt + sigma_dot * W
            target_m = dIdt - sigma_dot * W
            drift_p = model(model_in_p, t.float().view(-1, 1), **kwargs)
            drift_m = model(model_in_m, t.float().view(-1, 1), **kwargs)
            loss_p = 0.5 * self.image_sq_norm(drift_p - target_p).mean()
            loss_m = 0.5 * self.image_sq_norm(drift_m - target_m).mean()
            loss = loss_p + loss_m

        else:
            I_noised = I + sigma * W
            model_in = torch.cat([x, I_noised], dim=-1)
            drift = model(model_in, t.float().view(-1, 1), **kwargs)
            target = dIdt + sigma_dot * W

            loss= self.image_sq_norm(drift - target).mean()
        return loss

    def sample(self, x, model, refinement_steps=None, **kwargs):
        
        if refinement_steps is None:
            refinement_steps = self.num_refinement_steps

        timesteps = torch.linspace(0, 1, refinement_steps + 1, device=x.device) # shape (refinement_steps+1,)
        
        # start y at the source distribution after first step
        # We take 1st step analytically since g_T can be singular at t=0 during Euler-Maruyama integration
        input_0 = torch.cat([x, x], dim=-1)  # shape (b, nx, ny, 2d)
        sigma_0 = self.sigma(timesteps[0].expand(x.shape[0]), sample=True)  # shape (b, 1, 1, 1)
        noise_0 = self.get_noise(size=x.shape, device=x.device)
        dt = timesteps[1] - timesteps[0]
        drift_0 = model(input_0, timesteps[0].expand(x.shape[0]).unsqueeze(-1), **kwargs)
        dW = torch.sqrt(dt) * noise_0

        if self.method == "em":
            y = x + drift_0*dt + sigma_0 * dW # shape (b, nx, ny, d)
        else:
            y = x + drift_0*dt

        # integrate with euler-maruyama or euler method
        g_fn = lambda t: self.sigma(t, sample=True)
        y = self.integrator.integrate(x, y, model, timesteps[1:], g_fn, use_gF=self.use_gF, scheduler=self, **kwargs)

        return y

    def forward(self, x, y, model, **kwargs):
        return self.compute_loss(x, y, model, **kwargs)