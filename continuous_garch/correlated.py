
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import pyro.infer as infer
import pyro.infer.reparam as reparam
import pyro.optim as optim
import pyro.poutine as poutine
import torch
import torch.nn as nn
import torchsde
from pyro.infer.autoguide import (
    AutoDelta,
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    init_to_mean,
)
from pyro.nn.module import to_pyro_module_

# Correlation experiment - used to directly model correlation where possible
# Doesn't work super well except to handle the relationship between bias and
# models, as the correlation is in the ERROR term

def vec_to_tril_matrix(vec, diag=0):
    """
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    # +ve root of D**2 + (1+2*diag)*D - |diag| * (diag+1) - 2*vec.shape[-1] = 0
    n = (
        -(1 + 2 * diag)
        + ((1 + 2 * diag) ** 2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1)) ** 0.5
    ) / 2
    eps = torch.finfo(vec.dtype).eps
    if not torch._C._get_tracing_state() and (round(n) - n > eps):
        raise ValueError(
            f"The size of last dimension is {vec.shape[-1]} which cannot be expressed as "
            + "the lower triangular part of a square D x D matrix."
        )
    n = torch.round(n).long() if isinstance(n, torch.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + torch.Size((n, n)))
    arange = torch.arange(n, device=vec.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat


def to_corr(input_tensor):
    x = torch.tanh(input_tensor)
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(min=-1 + eps, max=1 - eps)
    r = vec_to_tril_matrix(x, diag=-1)
    z = r**2
    z1m_cumprod_sqrt = (1 - z).sqrt().cumprod(-1)
    # Diagonal elements must be 1.
    r = r + torch.eye(r.shape[-1], dtype=r.dtype, device=r.device)
    return r * torch.nn.functional.pad(z1m_cumprod_sqrt[..., :-1], [1, 0], value=1)

class GarchModel(pyro.nn.PyroModule):
    def __init__(self, p_mu=7, p_sigma=5, q_mu=3, q_sigma=2, ):
        super().__init__()
        self.p_mu, self.p_sigma = p_mu, p_sigma
        self.q_mu, self.q_sigma =  q_mu, q_sigma
        self.n_steps_ar = max(p_mu, q_mu)
        self.n_steps_ch = max(p_sigma, q_sigma)
        
        means = np.random.normal(0, 0.05, N_LAGS)
        sigmas = np.random.lognormal(-3, 0.25, N_LAGS)
        means[0] = np.abs(means[0]) + 0.07


    def forward(self, data:torch.Tensor):
        num_obs = data.size(0)
        # epsilon = torch.diff(data, dim=0)

        expectation = torch.zeros(num_obs, *data.shape[1:], dtype=data.dtype)
        resid = torch.zeros(num_obs, *data.shape[1:], dtype=data.dtype)
        expectation[: self.n_steps_ar] = data[: self.n_steps_ar]
        resid = expectation[: self.n_steps_ar] = torch.diff(data[: self.n_steps_ar+1], dim=0)
        ar_mu = pyro.sample("ar_mu", dist.Normal(0, 0.5).expand((num_obs-self.n_steps_ar,)).to_event(1))
        ar_a = pyro.sample("ar_a", dist.Normal(0, 0.5).expand((num_obs-self.n_steps_ar, self.p_mu,)).to_event(2))
        ar_b = pyro.sample("ar_b", dist.Normal(0, 0.5).expand((num_obs-self.n_steps_ar, self.q_mu,)).to_event(2))

        for idx in range(self.n_steps_ar, num_obs):
            ar_component = expectation[idx-self.p_mu:idx].T @ ar_a[idx]
            ma_component = resid[idx - self.q_mu:idx].T @ ar_b[idx]
            mu_hat = (ar_mu[idx] + ar_component + ma_component).view(1,-1)
            print(mu_hat.shape, data[idx].shape,(data[idx] - mu_hat).shape)
            expectation[idx], resid[idx] = mu_hat, (data[idx] - mu_hat)
            raise ValueError


        ch_z = pyro.sample(
            "ch_z", dist.Normal(0, 0.5).expand((num_obs, self.n_steps_ar)).to_event(2)
        )
        ch_sigma = pyro.sample(
            "ch_mu", dist.Normal(0, 0.5).expand((num_obs, self.n_steps_ar)).to_event(2)
        )

        ch_epsilon = residuals.sqrt() * ch_z
        
        expectations[: self.n_steps_ar] = data[: self.n_steps_ar]
        variances[: self.n_steps_ar] = residuals[: self.n_steps_ar]

        ar_coef_mean = pyro.sample(
            "ar_coed", dist.Normal(0, 0.5).expand((num_obs, self.n_steps_ar)).to_event(2)
        )

        noise_scale = pyro.sample("obs_scale", dist.LogNormal(-5.0, 4.0))
        phi = pyro.sample("ar_phi", dist.Normal(ar_coef_mean, ar_coef_std).to_event(2))

        for idx in range(num_obs -1):
            idx_start_ar = idx - self.n_steps_ar
            if idx_start_ar >= 0:
                means[idx+1] = (means[idx_start_ar:idx] * phi[idx]).sum()
        # ar_coef_mat = pyro.sample(
        #     "ar_corr", dist.LKJCholesky(concentration=2.0, dim=num_steps)
        # )
        scale_mat = ar_coef_std * torch.eye(num_steps)
        corr_scale = ar_coef_mat @ scale_mat
        phi = pyro.sample(
            "ar_phi", dist.MultivariateNormal(ar_coef_mean, scale_tril=corr_scale)
        )

        phi = phi.tile((num_obs, 1, 1))

        outputs = torch.bmm(phi, data.unsqueeze(-1)).squeeze(-1)
        with pyro.plate("output", num_obs, dim=-2):
            return pyro.sample(
                "obs",
                dist.Cauchy(
                    # stability=torch.sigmoid(noise_stability) + 1,
                    # skew=torch.tanh(noise_skew),
                    loc=0,
                    scale=noise_scale,
                ),
                obs=obs,
            )

    def guide(self, data, obs=None):
        _, num_steps = data.shape
        # Number of parameters needed to parametrise a correlation matrix.
        noise_loc = pyro.param("noise_loc", torch.tensor(-3.0))
        noise_scale = pyro.param(
            "noise_scale", torch.tensor(0.001), constraint=dist.constraints.positive
        )
        ar_mean_loc = pyro.param("ar_mean_loc", torch.zeros(num_steps))
        ar_mean_scale = pyro.param(
            "ar_mean_scale",
            torch.ones(num_steps) * 0.01,
            constraint=dist.constraints.positive,
        )
        ar_std_loc = pyro.param("ar_std_loc", torch.ones(num_steps) * -3)
        ar_std_scale = pyro.param(
            "ar_std_scale",
            torch.ones(num_steps) * 0.01,
            constraint=dist.constraints.positive,
        )

        ar_coef_mean = pyro.sample(
            "ar_mean", dist.Normal(ar_mean_loc, ar_mean_scale).to_event(1)
        )
        ar_coef_std = pyro.sample(
            "ar_std", dist.LogNormal(ar_std_loc, ar_std_scale).to_event(1)
        )
        pyro.sample("obs_scale", dist.LogNormal(noise_loc, noise_scale))
        pyro.sample("ar_phi", dist.Normal(ar_coef_mean, ar_coef_std).to_event(1))

        q = num_steps * (num_steps - 1) // 2
        cholesky_params_loc = pyro.param("cholesky_params_loc", torch.zeros(q))
        cholesky_params_scale = pyro.param(
            "cholesky_params_scale",
            torch.ones(q) * 0.01,
            constraint=dist.constraints.positive,
        )
        cholesky_params = pyro.sample(
            "cholesky_params",
            dist.Normal(cholesky_params_loc, cholesky_params_scale).to_event(1),
        )
        ar_coef_mat = pyro.sample(
            "ar_corr", dist.Delta(to_corr(cholesky_params)).to_event(2)
        )
        scale_mat = ar_coef_std * torch.eye(num_steps)
        corr_scale = ar_coef_mat @ scale_mat
        pyro.sample("ar_phi", dist.MultivariateNormal(ar_coef_mean, scale_tril=corr_scale))
