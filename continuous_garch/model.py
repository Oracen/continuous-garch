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


def summary(samples):
    # Helper function to get quantiles of an estimate
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).transpose()
        site_stats[site_name] = describe[
            ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
        ]
    return site_stats


def arma_garch_forward(
    num_obs,
    n_steps_ar,
    n_steps_ch,
    p_mu,
    q_mu,
    p_sigma,
    q_sigma,
    data,
    diffed_inputs,
    ar_mu,
    ar_a,
    ar_b,
    ch_omega,
    ch_alpha,
    ch_beta,
):
    # Seed values
    mu = []
    residual = []
    # Estimate ARMA process
    # for idx in pyro.markov(range(num_obs), n_steps_ar):
    diffed_inputs = diffed_inputs[..., : n_steps_ar + 1, :]
    for idx in range(num_obs):
        if idx < n_steps_ar:
            mu.append(data[..., idx : idx + 1, :])
            residual.append(diffed_inputs[..., idx : idx + 1, :])
            continue
        item = data[..., idx - p_mu : idx, :].detach()
        ar_component = item * ar_a[..., idx, :, :].transpose(-1, -2)
        ar_component = ar_component.mean(dim=-2, keepdim=True)
        local_resid = torch.cat(residual[idx - q_mu : idx], dim=-2)
        ma_component = local_resid * ar_b[..., idx, :, :].transpose(-1, -2)
        ma_component = ma_component.mean(dim=-2, keepdim=True)
        mu_hat = ar_mu[..., idx, :, :].transpose(-1, -2) + ar_component + ma_component
        mu.append(mu_hat)
        residual.append(item[..., -1:, :] - mu_hat)

    # Cat outputs
    mu = torch.cat(tuple(mu), dim=-2)
    residual = torch.cat(tuple(residual), dim=-2)
    resid_sq = residual.pow(2)

    # Init sigma + params
    sigma_sq = []

    # Estimate GARCH process
    # for idx in pyro.markov(range(num_obs), n_steps_ch):
    for idx in range(num_obs):
        if idx < n_steps_ch:
            sigma_sq.append(resid_sq[..., idx : idx + 1, :])
            continue
        item = resid_sq[..., idx - p_sigma : idx, :].detach()
        eps_component = item * ch_alpha[..., idx, :, :].transpose(-1, -2)
        eps_component = eps_component.mean(dim=-2, keepdim=True)
        local_resid = torch.cat(tuple(sigma_sq[idx - q_sigma : idx]), dim=-2)
        vol_component = local_resid * ch_beta[..., idx, :, :].transpose(-1, -2)
        vol_component = vol_component.mean(dim=-2, keepdim=True)
        sigma_sq.append(
            (
                ch_omega[..., idx, :, :].transpose(-1, -2)
                + eps_component
                + vol_component
            ).abs()
        )

    # Cat output
    sigma_sq = torch.cat(tuple(sigma_sq), dim=-2)

    # Constrain params
    sigma = sigma_sq.sqrt() + 1e-7
    latent_z = residual / sigma
    return mu, sigma, latent_z


class GarchModel(pyro.nn.PyroModule):
    def __init__(
        self,
        n_series=1,
        p_mu=7,
        p_sigma=5,
        q_mu=3,
        q_sigma=2,
    ):
        super().__init__()
        self.n_series = n_series
        self.p_mu, self.p_sigma = p_mu, p_sigma
        self.q_mu, self.q_sigma = q_mu, q_sigma
        self.n_steps_ar = max(p_mu, q_mu)
        self.n_steps_ch = max(p_sigma, q_sigma)
        self.n_obs = max(self.n_steps_ar, self.n_steps_ch)

    def forward(self, data: torch.Tensor):
        num_obs = data.size(-2)

        # Sample params
        ar_mu = pyro.sample(
            "ar_mu",
            dist.Normal(0, 0.25).expand((num_obs, self.n_series, 1)).to_event(3),
        )
        ar_a = pyro.sample(
            "ar_a",
            dist.Normal(0, 0.25)
            .expand((num_obs, self.n_series, self.p_mu))
            .to_event(3),
        )
        ar_b = pyro.sample(
            "ar_b",
            dist.Normal(0, 0.25)
            .expand((num_obs, self.n_series, self.q_mu))
            .to_event(3),
        )

        ch_omega = pyro.sample(
            "ch_omega",
            dist.Normal(0, 0.25).expand((num_obs, self.n_series, 1)).to_event(3),
        )
        ch_alpha = pyro.sample(
            "ch_alpha",
            dist.Normal(0, 0.25)
            .expand((num_obs, self.n_series, self.p_sigma))
            .to_event(3),
        )
        ch_beta = pyro.sample(
            "ch_beta",
            dist.Normal(0, 0.25)
            .expand((num_obs, self.n_series, self.q_sigma))
            .to_event(3),
        )
        data_size = list(ar_a.shape[:-3] + data.shape)
        data = data.expand(data_size)
        diffed_inputs = torch.diff(data, dim=-2)
        mu, sigma, latent_z = arma_garch_forward(
            num_obs,
            self.n_steps_ar,
            self.n_steps_ch,
            self.p_mu,
            self.q_mu,
            self.p_sigma,
            self.q_sigma,
            data,
            diffed_inputs,
            ar_mu,
            ar_a,
            ar_b,
            ch_omega,
            ch_alpha,
            ch_beta,
        )
        pyro.sample(
            "ch_z",
            dist.Normal(0, 1).expand((num_obs, self.n_series)).to_event(2),
            obs=latent_z,
        )
        pyro.deterministic("mu", mu)
        pyro.deterministic("sigma", sigma)
        cut = num_obs - self.n_obs
        data_size[-2] = cut
        pyro.sample(
            "obs_center",
            dist.SoftLaplace(mu[..., -cut:, :], 0.01).expand(data_size).to_event(2),
            obs=data[..., -cut:, :],
        )
        pyro.sample(
            "vol_center",
            dist.SoftLaplace(sigma[..., -cut:, :], 0.01).expand(data_size).to_event(2),
            obs=diffed_inputs[..., -cut:, :].abs(),
        )
        return pyro.sample(
            "obs",
            dist.SoftLaplace(mu[..., -cut:, :], sigma[..., -cut:, :])
            .expand(data_size)
            .to_event(2),
            obs=data[..., -cut:, :],
        )


class NNet(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hidden=32):
        super().__init__()
        self.lin = torch.nn.Linear(n_in, n_out)
        self.nlin1 = torch.nn.Linear(n_in, n_hidden)
        self.nlin2 = torch.nn.Linear(n_hidden, n_hidden)
        self.nlin3 = torch.nn.Linear(n_hidden, n_out)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        # nonlin_part = self.act(self.nlin3(self.act(self.nlin1(x))))
        nonlin_part = self.nlin3(self.act(self.nlin2(self.act(self.nlin1(x)))))
        return nonlin_part
        lin_part = self.lin(x)
        return lin_part + nonlin_part  # * (self.scale.exp() + 1e-7)


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, state_size):
        super().__init__()
        self.mu = NNet(state_size + 1, state_size)
        # self.sigma = torch.nn.Linear(state_size,
        #                              state_size * brownian_size)
        self.sigma = NNet(state_size + 1, state_size)
        # self.mean_scale = torch.nn.Parameter(torch.tensor(-2.0), True)
        self.std_scale = torch.nn.Parameter(torch.tensor(-3.0), True)
        # self.mu.weight.data *=
        # self.sigma.weight.data *= 0.001

    # Drift
    def f(self, t, y):
        scale_down = y.abs()
        # return self.mu(self.join_obs(y, t))# * self.mean_scale.exp()
        return -(torch.sign(y) * scale_down) + self.mu(self.join_obs(y, t))
        # return  -torch.sign(y) * self.mu(torch.tanh(y)) **2  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        base_std = self.sigma(self.join_obs(y, t))
        return (base_std).exp()

    def join_obs(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat((y, t[None, None].tile(y.size(0), 1)), dim=-1)


def format_sample(tensor: torch.Tensor, shape: torch.Size):
    shape = [tensor.size(0)] + shape + [-1, tensor.size(-1)]
    return tensor.reshape(shape).transpose(0, -3)


class GarchGuide(pyro.nn.PyroModule):
    def __init__(
        self,
        n_series=1,
        p_mu=7,
        p_sigma=5,
        q_mu=3,
        q_sigma=2,
    ):
        super().__init__()
        self.n_series = n_series
        self.p_mu, self.p_sigma = p_mu, p_sigma
        self.q_mu, self.q_sigma = q_mu, q_sigma
        self.n_steps_ar = max(p_mu, q_mu)
        self.n_steps_ch = max(p_sigma, q_sigma)
        self.n_obs = max(self.n_steps_ar, self.n_steps_ch)
        self.state_size = self.p_mu + self.q_mu + self.p_sigma + self.q_sigma + 2
        # self.y0 = pyro.param("y_0", torch.zeros(n_series, self.state_size))
        # I don't know if I should have made this a pyroparam but I assume it makes no
        # difference...pyroparams are just nnparams with some extra functionality
        self.y0 = torch.nn.Parameter(
            torch.randn(n_series, self.state_size) * 0.00001, True
        )
        self.y0_noise_scale = torch.nn.Parameter(torch.tensor(-5.0), True)
        self.sde = pyro.nn.PyroModule[SDE](self.state_size)
        # self.param_map = pyro.nn.PyroModule[NNet](1, self.state_size)

    def forward(self, data: torch.Tensor):
        plate_sizes = [
            item.size
            for item in poutine.runtime._PYRO_STACK
            if isinstance(item, pyro.plate)
        ]
        num_obs = data.size(-2)
        num_repeats = math.prod(plate_sizes)
        ts = torch.linspace(0, 1, num_obs)
        batched_y0 = self.y0.tile([num_repeats, 1])
        # Jitter the start location
        noise = torch.randn(batched_y0.shape) * (self.y0_noise_scale.exp() + 1e-7)
        batched_y0 = batched_y0 + noise
        # params will have shape (t_size, batch_size, state_size)
        params = torchsde.sdeint_adjoint(
            self.sde, batched_y0, ts, method="reversible_heun"
        )  # * self.y0_noise_scale.exp()

        # Uncomment param_map to use a raw nnet (non-stochastic), or y0 to just use a
        # deterministic single set of params
        # params = self.param_map(ts[...,None])[...,None,:].tile([1,num_repeats,1])
        # params = self.y0[ None].tile([num_obs, num_repeats, 1])
        ar_mu, ar_a, ar_b, ch_omega, ch_alpha, ch_beta = torch.split(
            params, [1, self.p_mu, self.q_mu, 1, self.p_sigma, self.q_sigma], dim=-1
        )
        pyro.sample("ar_mu", dist.Delta(format_sample(ar_mu, plate_sizes)).to_event(3))
        pyro.sample("ar_a", dist.Delta(format_sample(ar_a, plate_sizes)).to_event(3))
        pyro.sample("ar_b", dist.Delta(format_sample(ar_b, plate_sizes)).to_event(3))
        pyro.sample(
            "ch_omega", dist.Delta(format_sample(ch_omega, plate_sizes)).to_event(3)
        )
        pyro.sample(
            "ch_alpha", dist.Delta(format_sample(ch_alpha, plate_sizes)).to_event(3)
        )
        pyro.sample(
            "ch_beta", dist.Delta(format_sample(ch_beta, plate_sizes)).to_event(3)
        )
