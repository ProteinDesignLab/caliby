from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType


class EDM(nn.Module):
    def __init__(self, cfg: DictConfig, sigma_data: TensorType[(), float]):
        """
        EDM from Karras et al.

        Unlike Karras et al., time steps go from 0 (pure noise) to 1 (clean data)
        for consistency with other interpolants.
        """
        super().__init__()
        self.cfg = cfg

        self.register_buffer("sigma_data", sigma_data)  # set sigma_data
        self.rho = cfg.rho  # controls how large steps are at low noise are vs. high noise, higher prioritizes low noise
        self.s_min = cfg.s_min  # minimum noise level
        self.s_max = cfg.s_max  # maximum noise level

        # Training noise distribution
        self.training_noise_schedule = cfg.training_noise_schedule
        assert self.training_noise_schedule in [
            "lognormal",
            "uniform_sigma",
            "uniform_t",
            "trunc_normal_t",
            "constant_t",
        ], f"Unknown timestep schedule: {self.training_noise_schedule}"

        self.training_noise_cfg = cfg.training_noise_cfg[self.training_noise_schedule]

    @torch.compiler.disable
    def forward(
        self,
        batch: Dict[str, TensorType["b ..."]],
        t: Optional[TensorType["b", float]] = None,
    ) -> Dict[str, Any]:
        x1 = batch["x"]

        # Sample time steps if not provided
        if t is None:
            t = self.sample_timestep(x1.shape[0], device=x1.device)

        xt = self.noise_x(x1, t)

        # Construct outputs
        outputs = {}
        outputs["t"] = t  # [b]
        outputs["x_noised"] = xt
        outputs["x_target"] = x1  # we directly predict x1
        outputs["loss_weight_t"] = self.get_loss_weight(t)  # [b]

        return outputs

    def sigma(self, t: TensorType["b", float]) -> TensorType["b", float]:
        """
        Convert time step to noise level.
        """
        with torch.autocast(device_type=t.device.type, enabled=False):
            term1 = self.s_max ** (1 / self.rho)
            term2 = t * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            sigma = (term1 + term2) ** self.rho
            sigma = sigma * self.sigma_data
        return sigma

    def sigma_inv(self, sigma: TensorType["b", float]) -> TensorType["b", float]:
        """
        Convert noise level to time step.
        """
        with torch.autocast(device_type=sigma.device.type, enabled=False):
            sigma = sigma / self.sigma_data
            term_1_2 = sigma ** (1 / self.rho)
            term_1 = self.s_max ** (1 / self.rho)
            t = (term_1_2 - term_1) / (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
        return t

    def sample_timestep(self, n: int, device: torch.device) -> TensorType["b"]:
        """
        Sample a batch of b timesteps from the noise schedule.
        """
        if self.training_noise_schedule == "lognormal":
            log_sigmas = (
                torch.randn(n, device=device) * self.training_noise_cfg.psigma_std + self.training_noise_cfg.psigma_mean
            )
            sigmas = torch.exp(log_sigmas)
            sigmas = self.sigma_data * sigmas
            t = self.sigma_inv(sigmas)
        elif self.training_noise_schedule == "uniform_sigma":
            sigmas = torch.rand(n, device=device) * (self.s_max - self.s_min) + self.s_min
            sigmas = self.sigma_data * sigmas
            t = self.sigma_inv(sigmas)
        elif self.training_noise_schedule == "uniform_t":
            t_min, t_max = self.training_noise_cfg.t_min, self.training_noise_cfg.t_max
            t = torch.rand(n, device=device) * (t_max - t_min) + t_min
        elif self.training_noise_schedule == "trunc_normal_t":
            from scipy import stats

            loc, scale = self.training_noise_cfg.loc, self.training_noise_cfg.scale
            t_min, t_max = self.training_noise_cfg.t_min, self.training_noise_cfg.t_max
            a, b = (t_min - loc) / scale, (t_max - loc) / scale
            t = stats.truncnorm.rvs(a, b, loc=loc, scale=scale, size=n)
            t = torch.tensor(t, device=device, dtype=torch.float32)
        elif self.training_noise_schedule == "constant_t":
            t = torch.ones(n, device=device) * self.training_noise_cfg.t

        return t

    def sample_prior(self, shape: Tuple, device: torch.device) -> TensorType["b n a 3"]:
        """
        Sample n samples from the prior.
        """
        sigma = self.sigma(torch.zeros(shape[0], device=device))
        return torch.randn(*shape, device=device) * rearrange(sigma, "b -> b 1 1 1")

    def noise_x(self, x: TensorType["b n a 3"], t: TensorType["b"]) -> TensorType["b n a 3"]:
        """
        Add noise to x.
        """
        sigma = self.sigma(t)
        return x + torch.randn_like(x) * rearrange(sigma, "b -> b 1 1 1")

    def euler_step(
        self,
        f: Callable,
        xt: TensorType["b n a 3", float],
        t: TensorType["b", float],
        t_next: TensorType["b", float],
        step_scale: float,
    ) -> Tuple[
        TensorType["b n a 3", float],  # xt_next
        Dict[str, TensorType["b ..."]],  # aux preds
    ]:
        """
        Take an Euler step using the function f.

        f is the forward function of the denoiser trained with this interpolant.
        - It should take in the current state and the current time.
        """
        x1_pred, aux_preds = f(xt, t=t)
        aux_preds["x1_pred"] = x1_pred  # save x1_pred before any guidance modifications

        # Handle step scale
        score = (xt - x1_pred) / rearrange(self.sigma(t), "b -> b 1 1 1")
        score = score * step_scale

        dsigma = rearrange(self.sigma(t_next) - self.sigma(t), "b -> b 1 1 1")
        xt_next = xt + dsigma * score

        # Add to auxiliary outputs
        aux_preds["x1_pred"] = x1_pred

        return xt_next, aux_preds

    def get_loss_weight(self, t: TensorType["b"]) -> TensorType["b"]:
        """
        Compute the weight of the loss at time t.
        """
        c_out = self.c_out(self.sigma(t))
        return 1 / (c_out**2)

    def c_in(self, sigma: TensorType["b", float]) -> TensorType["b", float]:
        var_x = self.sigma_data**2 + sigma**2
        return 1 / torch.sqrt(var_x)

    def c_out(self, sigma: TensorType["b", float]) -> TensorType["b", float]:
        var_x = self.sigma_data**2 + sigma**2
        return sigma * self.sigma_data / torch.sqrt(var_x)

    def c_skip(self, sigma: TensorType["b", float]) -> TensorType["b", float]:
        var_x = self.sigma_data**2 + sigma**2
        return self.sigma_data**2 / var_x

    def c_noise(self, sigma: TensorType["b", float]) -> TensorType["b", float]:
        return 1 / 4 * torch.log(sigma)

    def setup_preconditioning(
        self,
        x_noised: TensorType["b n a 3", float],
        x_self_cond: Optional[TensorType["b n a 3", float]],
        t: TensorType["b", float],
    ) -> Tuple[Callable, Callable]:
        """
        Set up preconditioning input and output functions.
        """
        sigma = self.sigma(t)
        c_in = rearrange(self.c_in(sigma), "b -> b 1 1 1")
        c_noise = self.c_noise(sigma)
        c_skip = rearrange(self.c_skip(sigma), "b -> b 1 1 1")
        c_out = rearrange(self.c_out(sigma), "b -> b 1 1 1")

        def precondition_in() -> (
            Tuple[
                TensorType["b n a 3", float],  # x_noised
                TensorType["b n a 3", float],  # x_self_cond
                TensorType["b", float],  # c_noise
            ]
        ):
            pc_x_noised = x_noised * c_in

            pc_x_self_cond = None
            if x_self_cond is not None:
                pc_x_self_cond = x_self_cond / self.sigma_data  # scale to variance of 1

            return pc_x_noised, pc_x_self_cond, c_noise

        def precondition_out(
            denoiser_pred: Union[TensorType["b n a 3", float], TensorType["m b n a 3", float]],
        ) -> TensorType["b n a 3", float]:
            return c_skip * x_noised + c_out * denoiser_pred

        return precondition_in, precondition_out
