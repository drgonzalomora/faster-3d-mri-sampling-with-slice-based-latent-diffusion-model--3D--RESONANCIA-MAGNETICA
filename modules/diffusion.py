import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype = np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

class Diffusion(nn.Module):
    def __init__(
        self,
        T,
        beta_schedule,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        input_perturbation=0.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.input_perturbation = input_perturbation
        assert beta_schedule in ['linear', 'cosine'], "Beta schedule must be either 'linear' or 'cosine'"
        self.beta_schedule = beta_schedule
        self.T= T

        betas = linear_beta_schedule(T) if beta_schedule == 'linear' else cosine_beta_schedule(T)
        assert (betas > 0).all() and (betas < 1).all(), "Betas must be in ]0, 1["
        assert (betas[1:] - betas[:-1] > 0).all(), "Betas must be strictly increasing"
        self.betas = betas

        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # for q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1 / self.alphas_cumprod - 1)

        # for p(x_{t-1} | x_t})
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]) 
            # duplicate first element because posterior_variance is 0 at the beginning of the diffusion
        )
        
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1 - self.alphas_cumprod)

    def extract(self, arr, timestep, shape):
        arr = torch.from_numpy(arr).to(device=timestep.device, dtype=torch.float64)[timestep]
        return arr[(...,) +  (None,) * (shape.__len__() - arr.ndim)]
    
    def q_mean_variance(self, x_0, t):
        """get the distribution q(x_t | x_0) [the forward process]"""
        mean = x_0 * self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        variance = self.extract(1 - self.alphas_cumprod, t, x_0.shape) # variance, so no sqrt
        logvar = self.extract(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, logvar
    
    def q_sample(self, x_0, t, noise=None):
        """ sample from the distribution q(x_t | x_0) [the forward process directly from n steps]"""
        if noise is None:
            noise = torch.randn_like(x_0)
        mean, variance, _ = self.q_mean_variance(x_0, t)
        return mean + torch.sqrt(variance) * noise
    
    def q_posterior_mean_variance(self, x_t_prev, x_t, t):
        """get distribution q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_t_prev + 
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x_t, t, clamp=True):
        """get x_{t-1} ~ p(x_{t-1} | x_t) [the backward process]"""
        assert x_t.shape[0] == t.shape[0], "x_t and t must have the same batch size"
        output = model(x_t, t)
        
        # 1. Processing the variance
        ############################################################
        # TODO
        # 2. Processing the mean
        ############################################################
        if self.model_mean_type == 'epsilon':
            pred_x = self.predict_x_start_from_eps(x_t, t, output)
            if clamp:
                pred_x = torch.clamp(pred_x, -1, 1)
            output_mean, output_variance, output_logvar = self.q_posterior_mean_variance(pred_x, x_t, t)

        else:
            raise NotImplementedError()
        
        return output_mean, output_variance, output_logvar, pred_x, output
        
    # sampling from the diffusion
    def predict_x_start_from_eps(self, x_t, t, eps):
        return self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
            self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    
    def p_sample(self, model, x_t, t, clamp=True, **kwargs):
        """sample x_{t-1} ~ p(x_{t-1} | x_t) [the backward process directly from n steps]"""
        mean, variance, _, _, _ = self.p_mean_variance(model, x_t, t, clamp=clamp)
        x_t_prev = mean + torch.sqrt(variance) * (torch.randn_like(mean) if t > 0 else torch.zeros_like(mean))
        return x_t_prev
    
    def ddim_p_sample(self, model, x_t, t, clamp=True, eta=0.0, **kwargs):
        """Sampling x_{t-1} according the DDIM scheme"""
        mean, variance, _, x_pred, eps = self.p_mean_variance(model, x_t, t, clamp=clamp)
        noise = torch.randn_like(x_t)

        alpha_cumprod_t = self.extract(self.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t_prev = self.extract(self.alphas_cumprod_prev, t, x_t.shape)
        sigma = (
            torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t))
            * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            * eta
        )

        x_t_prev = x_pred * torch.sqrt(alpha_cumprod_t_prev)
        x_t_prev += torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * eps
        x_t_prev = x_t_prev + sigma * (noise if t > 0 else 0)
        return x_t_prev
    
    def ddim_p_reverse_sample(self, model, x_t, t, eta=0.0, clamp=True):
        """Sample x_{t+1} from the model using DDIM reverse ODE"""
        assert eta == 0.0, "eta must be 0.0 for reverse sampling"
        mean, variance, _, x_pred, eps = self.p_mean_variance(model, x_t, t, clamp=clamp)

        alpha_cumprod_next = self.extract(self.alphas_cumprod_next, t, x_t.shape)
        x_t_next = x_pred * torch.sqrt(alpha_cumprod_next) + torch.sqrt(1 - alpha_cumprod_next) * eps
        return x_t_next
    
    def sample(self, model, noise, ddim=False, eta=0.0, clamp=True, verbose=True, intermediate=False):
        """Diffuse new samples"""
        assert noise is not None, "noise must be provided"
        B = noise.shape[0]
        x_t = noise

        if intermediate: intermediate_samples = []

        sample = self.ddim_p_sample if ddim else self.p_sample
        display_fn = tqdm if verbose else lambda x, **kwargs: x
        
        for timestep in display_fn(range(self.T - 1, -1, -1), desc='Sampling', position=0, leave=True):
            with torch.no_grad():
                t = torch.full(size=(B,), fill_value=timestep, dtype=torch.long, device=noise.device)
                x_t = sample(model, x_t, t, clamp=clamp, eta=eta)
                if intermediate:
                    intermediate_samples.append(x_t)
        
        if intermediate:
            return x_t, intermediate_samples
        return x_t
    
    def forward_step(self, model, x_0, t, noise=None, weights=None):
        """A single diffusion forward step, returns all required losses for training"""
        if noise is None:
            noise = torch.randn_like(x_0)

        noise = noise + self.input_perturbation * torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise).type(model.precision)

        # predict noise
        output = model(x_t, t)

        # TODO: process var types

        loss = F.mse_loss(output, noise, reduction='none')

        if weights is not None:
            loss = loss * weights[(...,) +  (None,) * (loss.ndim - 1)]

        return loss.mean().to(dtype=torch.float32)

class SimpleDiffusion(nn.Module):
    def __init__(self, noise_shape, T=1000, beta_schedule='cosine') -> None:
        super().__init__()
        assert beta_schedule in ['linear', 'cosine'], 'beta_schedule must be either linear or cosine'
        self.noise_shape = noise_shape
        self.T = T
        self.betas = self.cosine_beta_schedule(T) if beta_schedule == 'cosine' else self.linear_beta_schedule(T)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    def forward_step(self, module, images, times, noise=None, **kwargs):
        if noise is None:
            eps = torch.randn_like(images).to(images.device) 
        else:
            eps = noise

        gamma = eps + 0.1 * torch.randn_like(images).to(images.device)
        alpha_hat = self.alphas_hat[(times,) + (None,) * len(self.noise_shape)]

        x_t = torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gamma
        x_t = x_t.to(images.device)

        eps_hat = self.reverse_process(module, x_t, times)
        
        # loss
        loss = F.mse_loss(eps_hat, eps)
        return loss

        # return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gamma, eps
    
    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

    def cosine_beta_schedule(self, timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def reverse_process(self, model, x_t, time):
        return model(x_t, time)
    
    @torch.no_grad()
    def sample(self, model, x_T=None, n_sample=1, **kwargs):
        x_t = x_T
        if x_t is None:
            x_t = torch.randn(n_sample, *self.noise_shape).to(x_T.device)

        for timestep in tqdm(range(self.T - 1, -1, -1), desc='Sampling', position=0, leave=True):
            times = torch.full(size=(x_t.shape[0],), fill_value=timestep, dtype=torch.long, device=x_t.device)
            eps = self.reverse_process(model, x_t, times)
            beta_t = self.betas[timestep].to(x_t.device)
            alpha_t = self.alphas[timestep].to(x_t.device)
            alpha_hat_t = self.alphas_hat[timestep].to(x_t.device)
            alpha_hat_t_prev = self.alphas_hat[timestep - 1].to(x_t.device)
            beta_t_hat = (1 - alpha_hat_t_prev) / (1 - alpha_hat_t) * beta_t
            if timestep > 0:
                var = torch.sqrt(beta_t_hat) * torch.randn_like(x_t).to(x_t.device)
            else :
                var = 0
            x_t = alpha_t.rsqrt() * (x_t - beta_t / torch.sqrt((1 - alpha_hat_t_prev)) * eps) + var
        return x_t

class SpacedDiffusion(Diffusion):
    """Partially adapted from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py"""
    def __init__(self, 
        T, 
        beta_schedule, 
        model_mean_type, 
        model_var_type, 
        loss_type, 
        rescale_timesteps=False, 
        input_perturbation=0, 
        **kwargs
    ) -> None:
        super().__init__(
            T, 
            beta_schedule, 
            model_mean_type, 
            model_var_type, 
            loss_type, 
            rescale_timesteps, 
            input_perturbation,
            **kwargs
        )
        self.spaced_timesteps = self.get_spaced_timesteps(T, beta_schedule, rescale_timesteps)
        self.extra = self.T % self.spaced_timesteps.__len__()
        
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in self.spaced_timesteps:
                new_betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        self.betas = np.array(new_betas, dtype=np.float64)
            
    def get_spaced_timesteps(self, steps):
        """Compute the timesteps for the spaced diffusion"""
        stride = self.T // steps
        if stride <= 1:
            raise ValueError(f"Number of steps {steps} is too large for the number of timesteps {self.T} (use at least 2)")
        return np.arange(0, self.T, stride)

    