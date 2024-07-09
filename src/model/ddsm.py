"""
The file contains everything related to Dirichlet diffusion process such as generation noise code,
sampler code, likelihood computation code, and so on.
"""
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Transform, constraints
from torch.nn.functional import pad
from torch.autograd import grad
from torch.distributions.beta import Beta, Dirichlet
from torch.distributions.beta import Beta

from scipy import integrate

import tqdm
from numbers import Real


def beta_logp(alpha, beta, x):
    if isinstance(alpha, Real) and isinstance(beta, Real):
        concentration = torch.tensor([float(alpha), float(beta)])
    else:
        concentration = torch.stack([alpha, beta], -1)
    return dirichlet_logp(concentration, x)


def dirichlet_logp(concentration, x):
    x = torch.stack([x, 1.0 - x], -1)
    return (
            (torch.log(x) * (concentration - 1.0)).sum(-1)
            + torch.lgamma(concentration.sum(-1))
            - torch.lgamma(concentration).sum(-1)
    )


def log_rising_factorial(a, n):
    return torch.lgamma(a + n) - torch.lgamma(a)


def jacobi(x, alpha, beta, order=100):
    """
    Compute Jacobi polynomials.
    """
    a = alpha
    b = beta
    recur_fun = lambda p_n, p_n_minus1, n, x: (
                                                      torch.mul(
                                                          x * (2 * n + a + b + 2) * (2 * n + a + b) + (a ** 2 - b ** 2),
                                                          p_n)
                                                      * (2 * n + a + b + 1)
                                                      - p_n_minus1 * (n + a) * (n + b) * (2 * n + a + b + 2) * 2
                                              ) / (2 * (n + 1) * (n + a + b + 1) * (2 * n + a + b))

    if order == 0:
        return torch.ones_like(x)[:, None]
    else:
        ys = [torch.ones_like(x), (a + 1) + (a + b + 2) * (x - 1) * 0.5]
        for i in range(1, order - 1):
            ys.append(recur_fun(ys[i], ys[i - 1], i, x))
        return torch.stack(ys, -1)


def jacobi_diffusion_density(x0, xt, t, a, b, order=100, speed_balanced=True):
    """
    Compute Jacobi diffusion transition density function.
    """
    n = torch.arange(order, device=x0.device).double().expand(*x0.shape, order)
    if speed_balanced:
        s = 2 / (a + b)
    else:
        s = torch.ones_like(a)
    eigenvalues = (
            -0.5 * s.unsqueeze(-1) * n * (n - 1 + a.unsqueeze(-1) + b.unsqueeze(-1))
    )

    logdn = (
            log_rising_factorial(a.unsqueeze(-1), n)
            + log_rising_factorial(b.unsqueeze(-1), n)
            - log_rising_factorial((a + b).unsqueeze(-1), n - 1)
            - torch.log(2 * n + (a + b).unsqueeze(-1) - 1)
            - torch.lgamma(n + 1)
    )

    return (
            torch.exp(beta_logp(a, b, xt).unsqueeze(-1) + (eigenvalues * t - logdn))
            * jacobi(x0 * 2 - 1, alpha=b - 1, beta=a - 1, order=order)
            * jacobi(xt * 2 - 1, alpha=b - 1, beta=a - 1, order=order)
    ).sum(-1)


def Jacobi_Euler_Maruyama_sampler(
        x0, a, b, t, num_steps, speed_balanced=True, device="cuda", eps=1e-5
):
    """
    Generate Jacobi diffusion samples with the Euler-Maruyama solver.
    """
    a = a.to(device)
    b = b.to(device)
    x0 = x0.to(device)
    if speed_balanced:
        s = 2 / (a + b)
    else:
        s = torch.ones_like(a)

    def step(x, step_size):
        g = torch.sqrt(s * x * (1 - x))
        return (
                x
                + 0.5 * s * (a * (1 - x) - b * x) * step_size
                + torch.sqrt(step_size) * g * torch.randn_like(x)
        )

    time_steps = torch.linspace(0, t, num_steps, device=device)
    step_size = time_steps[1] - time_steps[0]
    x = x0
    with torch.no_grad():
        for _ in time_steps:
            x_next = step(x, step_size)
            x_next = x_next.clip(eps, 1 - eps)
            x = x_next
    return x


def noise_factory(N, n_time_steps, a, b, total_time=4, order=100,
                  time_steps=1000, speed_balanced=True, logspace=False,
                  mode="independent", device="cuda"):
    """
    Generate Jacobi diffusion samples and compute score of transition density function.
    """
    assert a.size() == b.size()
    noise_factory_one = torch.ones(N, n_time_steps, a.size(-1))
    noise_factory_zero = torch.zeros(N, n_time_steps, a.size(-1))

    if logspace:
        timepoints = np.logspace(np.log10(0.01), np.log10(total_time), n_time_steps)
    else:
        timepoints = np.linspace(0, total_time, n_time_steps + 1)[1:]

    if mode == "independent":
        for i, t in enumerate(timepoints):
            noise_factory_one[:, i, :] = Jacobi_Euler_Maruyama_sampler(
                noise_factory_one[:, i, :], a, b, t, time_steps,
                speed_balanced=speed_balanced, device=device)
            noise_factory_zero[:, i, :] = Jacobi_Euler_Maruyama_sampler(
                noise_factory_zero[:, i, :], a, b, t, time_steps,
                speed_balanced=speed_balanced, device=device)
    elif mode == "path":
        for i, t in enumerate(timepoints):
            if i == 0:
                noise_factory_one[:, i, :] = Jacobi_Euler_Maruyama_sampler(
                    noise_factory_one[:, i, :], a, b, timepoints[i], time_steps,
                    speed_balanced=speed_balanced, device=device)
                noise_factory_zero[:, i, :] = Jacobi_Euler_Maruyama_sampler(
                    noise_factory_zero[:, i, :], a, b, timepoints[i], time_steps,
                    speed_balanced=speed_balanced, device=device)
            else:
                noise_factory_one[:, i, :] = Jacobi_Euler_Maruyama_sampler(
                    noise_factory_one[:, i - 1, :],
                    a, b, timepoints[i] - timepoints[i - 1], time_steps,
                    speed_balanced=speed_balanced, device=device)
                noise_factory_zero[:, i, :] = Jacobi_Euler_Maruyama_sampler(
                    noise_factory_zero[:, i - 1, :],
                    a, b, timepoints[i] - timepoints[i - 1], time_steps,
                    speed_balanced=speed_balanced, device=device)
    else:
        raise ValueError

    noise_factory_one_loggrad = torch.zeros(N, n_time_steps, a.size(-1))
    noise_factory_zero_loggrad = torch.zeros(N, n_time_steps, a.size(-1))

    for i, t in enumerate(timepoints):
        xt = noise_factory_one[:, i, :].detach().clone()

        xt.requires_grad = True
        p = jacobi_diffusion_density(
            torch.ones(xt.size(), device=device),
            xt.to(device), t, a.to(device), b.to(device), order=order,
            speed_balanced=speed_balanced,
        )
        p.log().sum().backward()
        noise_factory_one_loggrad[:, i, :] = xt.grad.to(
            noise_factory_zero_loggrad.device
        )

        xt = noise_factory_zero[:, i, :].detach().clone()
        xt.requires_grad = True
        p = jacobi_diffusion_density(
            torch.zeros(xt.size(), device=device),
            xt.to(device), t, a.to(device), b.to(device),
            order=order,
            speed_balanced=speed_balanced,
        )
        p.log().sum().backward()
        noise_factory_zero_loggrad[:, i, :] = xt.grad.to(
            noise_factory_zero_loggrad.device
        )
    return (
        noise_factory_one,
        noise_factory_zero,
        noise_factory_one_loggrad,
        noise_factory_zero_loggrad,
        timepoints,
    )


class UnitStickBreakingTransform(Transform):
    """
    Transform from unconstrained 0-1 space to the simplex of one additional
    dimension via a stick-breaking process. This different from PyTorch's
    StickBreakingTransform which transform from unconstrained space and apply
    a sigmoid transform.
    This transform arises from an iterative stick-breaking
    construction of the `Dirichlet` distribution: the first probability p1 is
    used as is and second probability is multiplied by 1 - p1,
    the third probability is multiplied by (1- p1 -p2 ) and the process iterative.
    This is bijective and appropriate for use in HMC; however it mixes
    coordinates together and is less appropriate for optimization.
    """

    domain = constraints.unit_interval
    codomain = constraints.simplex
    bijective = True

    def __eq__(self, other):
        return isinstance(other, UnitStickBreakingTransform)

    def _call(self, x):
        x_cumprod = (1 - x).cumprod(-1)
        y = pad(x, [0, 1], value=1) * pad(x_cumprod, [1, 0], value=1)
        return y

    def _inverse(self, y, prevent_nan=False):
        y_crop = y[..., :-1]
        # sf = 1 - y_crop.cumsum(-1)
        # sacrifice some performance for better numerical stability
        sf = torch.flip(torch.flip(y, dims=(-1,)).cumsum(-1), dims=(-1,))[..., :-1]
        # inverse of 1
        if prevent_nan:
            x = y_crop / (sf + torch.finfo(y.dtype).tiny)
        else:
            x = y_crop / sf
        return x

    def log_abs_det_jacobian(self, x, y=None):
        # log det of the inverse transform
        detJ = -(
                (1 - x).log() * torch.arange(x.shape[-1] - 1, -1, -1, device=x.device)
        ).sum(-1)
        return detJ

    def log_abs_det_jacobian_forward(self, x, y=None):
        # log det of the forward transform
        detJ = (
                (1 - x).log() * torch.arange(x.shape[-1] - 1, -1, -1, device=x.device)
        ).sum(-1)
        return detJ

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] + 1,)

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] - 1,)


def simplex_diffusion_density(x0, xt, t, a, b, speed_balanced=True, order=100):
    sb = UnitStickBreakingTransform()
    y0 = sb._inverse(x0, prevent_nan=True)
    yt = sb._inverse(xt, prevent_nan=True)
    pt = jacobi_diffusion_density(
        y0, yt, t, a, b, order=order, speed_balanced=speed_balanced
    )
    return pt.log().sum(-1) + sb.log_abs_det_jacobian(xt, yt)


def gx_to_gv(gx, x, create_graph=False, compute_gradlogdet=True):
    gx = gx.double()
    x = x.double()
    sb = UnitStickBreakingTransform()
    v = sb._inverse(x, prevent_nan=True).detach()
    v.requires_grad = True
    x = sb(v)

    gv = grad(x, v, gx, create_graph=create_graph)[0]
    if compute_gradlogdet:
        logdet = sb.log_abs_det_jacobian(v, x)
        gradlogdet = grad(logdet.sum(), v, create_graph=create_graph)[0]
        gv = gv - gradlogdet
    return gv.float()


def gv_to_gx(gv, v, create_graph=False, compute_gradlogdet=True):
    gv = gv.double()
    v = v.double()
    sb = UnitStickBreakingTransform()
    x = sb(v).detach()
    x.requires_grad = True
    v = sb._inverse(x, prevent_nan=True)
    if compute_gradlogdet:
        logdet = sb.log_abs_det_jacobian(v, x)
        gradlogdet = grad(logdet.sum(), v)[0]
        return grad(v, x, gv + gradlogdet, create_graph=create_graph)[0].float()
    else:
        return grad(v, x, gv, create_graph=create_graph)[0].float()


#############################################
## DIFFUSION FACTORIES - FAST and NON-FAST ##
#############################################
def diffusion_factory(
        x, time_ind, noise_factory_one, noise_factory_zero,
        noise_factory_one_loggrad, noise_factory_zero_loggrad,
        alpha=None, beta=None, device="cuda", return_v=False, eps=1e-5, ):
    """
    Generate multivariate Jacobi diffusion samples and scores
    by sampling from noise factory for k-1 Jacobi diffusion processes.
    """
    time_ind = time_ind[(...,) + (None,) * (x.ndim - 2)].expand(x.shape[:-1])
    K = x.shape[-1]
    if alpha is None:
        alpha = torch.ones(K - 1)
    if beta is None:
        beta = torch.arange(K - 1, 0, -1, dtype=torch.float)

    noise_factory_size = noise_factory_one.shape[0]
    sb = UnitStickBreakingTransform()

    sample_inds = torch.randint(0, noise_factory_size, size=x.size()[:-1])
    v_samples = noise_factory_zero[sample_inds, time_ind, :].to(device).float()
    v_samples_grad = (
        noise_factory_zero_loggrad[sample_inds, time_ind, :].to(device).float()
    )

    inds = x == 1
    for i in range(K - 1):
        if torch.sum(inds[..., i]) != 0:
            v_samples[..., i][inds[..., i]] = (
                noise_factory_one[sample_inds[inds[..., i]], time_ind[inds[..., i]], i]
                    .to(device).float()
            )
            v_samples_grad[..., i][inds[..., i]] = (
                noise_factory_one_loggrad[
                    sample_inds[inds[..., i]], time_ind[inds[..., i]], i
                ].to(device).float()
            )
            if i + 1 < alpha.shape[0]:
                B = Beta(
                    alpha[i + 1:].float().to(device), beta[i + 1:].float().to(device)
                )
                v = B.sample((inds[..., i].sum(),)).clip(eps, 1 - eps)
                v = v.detach()
                v.requires_grad = True
                B.log_prob(v).sum().backward()
                v_samples[..., i + 1:][inds[..., i]] = v.detach()
                v_samples_grad[..., i + 1:][inds[..., i]] = v.grad

    if return_v:
        return v_samples, v_samples_grad
    else:
        v_samples.requires_grad = True
        samples = sb(v_samples)
        samples_grad = gv_to_gx(v_samples_grad, v_samples)
        samples_grad -= samples_grad.mean(-1, keepdims=True)
        return samples, samples_grad


def diffusion_fast_flatdirichlet(
        x, time_inds, noise_factory_one, noise_factory_one_loggrad, symmetrize=False
):
    """
    Fast multivariate Jacobi diffusion sampling assuming the stationary
    distribution is specified as Dir(1,1,1,1,...). Only requires
    noise factory for for one Jacobi diffusion process (1, k-1).
    """

    with torch.no_grad():
        k = x.shape[-1]
        sample_inds = torch.randint(0, noise_factory_one.shape[0], x.size()[:-1])
        x_samples = torch.zeros(x.size()).to(x.device)
        x_samples_grad = torch.zeros(x.size()).to(x.device)
        x_samples[..., 0] = noise_factory_one[
            sample_inds, time_inds[(...,) + (None,) * (x.ndim - 2)], 0]
        x_samples_grad[..., 0] = noise_factory_one_loggrad[
                                     sample_inds, time_inds[(...,) + (None,) * (x.ndim - 2)], 0
                                 ] + (k - 2) / (1 - x_samples[..., 0])

        D = Dirichlet(torch.ones(k - 1))
        d = D.sample((x.size()[:-1]))

        x_samples[..., 1:] = (1 - x_samples[..., [0]]) * d.data

        inds = x == 1
        if symmetrize:
            x_samples_grad -= x_samples_grad.mean(-1, keepdims=True)

        for i in range(k):
            if i != 0:
                temp = x_samples[..., i][inds[..., i]].detach().clone()
                x_samples[..., i][inds[..., i]] = x_samples[..., 0][inds[..., i]]
                x_samples[..., 0][inds[..., i]] = temp
                temp = x_samples_grad[..., i][inds[..., i]].detach().clone()
                x_samples_grad[..., i][inds[..., i]] = x_samples_grad[..., 0][
                    inds[..., i]
                ]
                x_samples_grad[..., 0][inds[..., i]] = temp

        return x_samples, x_samples_grad


#############################################
################ MODEL PART #################
#############################################
class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


#############################################
############### Samplers ####################
#############################################

def Euler_Maruyama_sampler(
        score_model,
        sample_shape,
        new_class = None,
        class_number = None,
        strength = None, 
        init=None,
        mask=None,
        alpha=None,
        beta=None,
        max_time=4,
        min_time=0.01,
        time_dilation=1,
        time_dilation_start_time=None,
        batch_size=64,
        num_steps=100,
        device="cuda",
        random_order=False,
        speed_balanced=True,
        speed_factor=None,
        concat_input=None,
        eps=1e-5,
):
    """
    Generate samples from score-based models with the Euler-Maruyama solver
    for (multivariate) Jacobi diffusion processes with stick-breaking
    construction.

    Parameters
    ----------
    score_model : torch.nn.Module
        A PyTorch time-dependent score model.
    sample_shape : tuple
        Shape of all dimensions of sample tensor without the batch dimension.
    init: torch.Tensor, default is None
        If specified, use as initial values instead of sampling from stationary distribution.
    alpha :  torch.Tensor, default is None
        Jacobi Diffusion parameters. If None, use default choices of alpha, beta =
        (1, k-1), (1, k-2), (1, k-3), ..., (1, 1) where k is the number of categories.
    beta : torch.Tensor, default is None
        See above `for alpha`.
    max_time : float, default is 4
        Max time of reverse diffusion sampling.
    min_time : float, default is 0.01
        Min time of reverse diffusion sampling.
    time_dilation : float, default is 1
        Use `time_dilation > 1` to bias samples toward high density areas.
    time_dilation_start_time : float, default is None
        If specified, start time dilation from this timepoint to min_time.
    batch_size : int, default is 64
        Number of samples to generate
    num_steps: int, default is 100
        Total number of steps for reverse diffusion sampling.
    device: str, default is 'cuda'
        Use 'cuda' to run on GPU or 'cpu' to run on CPU
    random_order : bool, default is False
        Whether to convert x to v space with randomly ordered stick-breaking transform.
    speed_balanced : bool, default is True
        If True use speed factor `s=(a+b)/2`, otherwise use `s=1`.
    eps: float, default is 1e-5
        All state values are clamped to (eps, 1-eps) for numerical stability.


    Returns
    -------
    Samples : torch.Tensor
        Samples in x space.
    """
 
    sb = UnitStickBreakingTransform()
    if alpha is None:
        alpha = torch.ones(sample_shape[-1] - 1, dtype=torch.float, device=device)
    if beta is None:
        beta = torch.arange(
            sample_shape[-1] - 1, 0, -1, dtype=torch.float, device=device
        )

    if speed_balanced:
        if speed_factor is None:
            s = 2.0 / (alpha + beta)
        else:
            s = speed_factor * 2.0 / (alpha + beta)
    else:
        s = torch.ones(sample_shape[-1] - 1).to(device)
  
    if init is None:
        init_v = Beta(alpha, beta).sample((batch_size,) + sample_shape[:-1]).to(device)
    else:
        init_v = sb._inverse(init).to(device)

    if time_dilation_start_time is None:
        time_steps = torch.linspace(
            max_time, min_time, num_steps * time_dilation + 1, device=device
        )
    else:
        time_steps = torch.cat(
            [
                torch.linspace(
                    max_time,
                    time_dilation_start_time,
                    round(num_steps * (max_time - time_dilation_start_time) / max_time)
                    + 1,
                )[:-1],
                torch.linspace(
                    time_dilation_start_time,
                    min_time,
                    round(num_steps * (time_dilation_start_time - min_time) / max_time)
                    * time_dilation
                    + 1,
                ),
            ]
        )
   
    step_sizes = time_steps[:-1] - time_steps[1:]
    time_steps = time_steps[:-1]
    v = init_v.detach()

    if mask is not None:
        assert mask.shape[-1] == v.shape[-1]+1

    if random_order:
        order = np.arange(sample_shape[-1])
    else:
        if mask is not None:
            mask_v = sb.inv(mask)

    with torch.no_grad():
        for i_step in tqdm.tqdm(range(len(time_steps))):
            time_step = time_steps[i_step]
            step_size = step_sizes[i_step]
            x = sb(v)

            if time_dilation_start_time is not None:
                if time_step < time_dilation_start_time:
                    c = time_dilation
                else:
                    c = 1
            else:
                c = time_dilation

            if not random_order:
                g = torch.sqrt(v * (1 - v))
                batch_time_step = torch.ones(batch_size, device=device) * time_step

                with torch.enable_grad():

                    if (concat_input is None) and (new_class is None):
                        score = score_model(x, batch_time_step, None)
                    else:
                        zero_class = class_number * torch.ones(batch_size)
                        zero_class = zero_class.type(torch.LongTensor)
                        zero_class = zero_class.to(device)
                        score = strength * score_model(x, batch_time_step, new_class) + (1.0 - strength) * score_model(x, batch_time_step, zero_class) 
                    #else:
                    #    score = score_model(torch.cat([x, concat_input], -1), batch_time_step)

                    mean_v = (
                            v + s[(None,) * (v.ndim - 1)] * (
                            (0.5 * (alpha[(None,) * (v.ndim - 1)] * (1 - v)
                                    - beta[(None,) * (v.ndim - 1)] * v)) - (1 - 2 * v)
                            - (g ** 2) * gx_to_gv(score, x)
                    ) * (-step_size) * c
                    )

                next_v = mean_v + torch.sqrt(step_size * c) * \
                         torch.sqrt(s[(None,) * (v.ndim - 1)]) * g * torch.randn_like(v)

                if mask is not None:
                    next_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]

                v = torch.clamp(next_v, eps, 1 - eps).detach()
            else:
                x = x[..., np.argsort(order)]
                order = np.random.permutation(np.arange(sample_shape[-1]))

                if mask is not None:
                    mask_v = sb.inv(mask[..., order])

                v = sb._inverse(x[..., order], prevent_nan=True)
                v = torch.clamp(v, eps, 1 - eps).detach()

                g = torch.sqrt(v * (1 - v))
                batch_time_step = torch.ones(batch_size, device=device) * time_step

                with torch.enable_grad():
                    if concat_input is None:
                        score = score_model(x, batch_time_step)
                    else:
                        score = score_model(torch.cat([x, concat_input], -1), batch_time_step)
                    mean_v = (v + s[(None,) * (v.ndim - 1)] * (
                            (0.5 * (alpha[(None,) * (v.ndim - 1)] * (1 - v)
                                    - beta[(None,) * (v.ndim - 1)] * v))
                            - (1 - 2 * v) - (g ** 2) * (gx_to_gv(
                        score[..., order],
                        x[..., order]))
                    ) * (-step_size) * c
                              )
                next_v = mean_v + torch.sqrt(step_size * c) * torch.sqrt(
                    s[(None,) * (v.ndim - 1)]
                ) * g * torch.randn_like(v)

                if mask is not None:
                    next_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]

                v = torch.clamp(next_v, eps, 1 - eps).detach()

    if mask is not None:
        mean_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]

    # Do not include any noise in the last sampling step.
    if not random_order:
        return sb(torch.clamp(mean_v, eps, 1 - eps))
    else:
        return sb(torch.clamp(mean_v, eps, 1 - eps))[..., np.argsort(order)]




  
#############################################
########## Likelihood estimations ###########
#############################################
def prior_likelihood(v, alpha, beta, device='cuda'):
    alpha = alpha.to(device)
    beta = beta.to(device)
    v = v.to(device)
    return beta_logp(alpha, beta, v).sum(dim=tuple(range(1, v.ndim)))


def ode_likelihood(v,
                   score_model,
                   max_time=4,
                   min_time=1e-2,
                   time_dilation=1,
                   device='cuda',
                   eps=1e-6,
                   alpha=None,
                   beta=None,
                   speed_balanced=True,
                   concat_input=None,
                   verbose=False):
    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    shape = v.shape
    epsilon = torch.randn(shape).to(device)

    if alpha is None:
        alpha = torch.ones(shape[-1])
    if beta is None:
        beta = torch.arange(shape[-1], 0, -1, dtype=torch.float)

    alpha = alpha.to(device)
    beta = beta.to(device)

    if speed_balanced:
        s = 2. / (alpha + beta)
    else:
        s = torch.ones(shape[-1]).to(device)

    sb = UnitStickBreakingTransform()

    def logodds(v):
        return np.log(np.fmax(v, eps) / np.fmax(1 - v, eps))

    def inverse_logodds(lov):
        return 1 - 1 / (1 + np.exp(lov))

    def divergence_eval_wrapper(sample, time_steps):
        # epsilon = torch.randn(shape).to(device)
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.enable_grad():
            sample.requires_grad_(True)
            x = sb(sample)
            g = torch.sqrt(sample * (1 - sample))
            logdet = sb.log_abs_det_jacobian(sample)
            gradlogdet = grad(logdet.sum(), sample, create_graph=True)[0]

            if concat_input is not None:
                score_x = score_model(torch.cat([x, concat_input], -1), time_steps * time_dilation)
            else:
                score_x = score_model(x, time_steps * time_dilation)
            score_v = grad(x, sample, score_x, create_graph=True)[0] - gradlogdet

            f_tilde = s[(None,) * (sample.ndim - 1)] * (0.5 * (alpha[(None,) * (sample.ndim - 1)] * (1 - sample) - beta[
                (None,) * (sample.ndim - 1)] * sample) - 0.5 * (1 - 2 * sample) - 0.5 * (g ** 2) * (score_v))
            score_e = torch.sum(f_tilde * epsilon)
            grad_score_e = grad(score_e, sample)[0]
        div = torch.sum(grad_score_e * epsilon, dim=tuple(range(1, grad_score_e.ndim))).cpu().numpy().astype(np.float64)
        return div, f_tilde.cpu().detach().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, ode_x):
        time_steps = np.ones((shape[0],)) * t
        v = inverse_logodds(ode_x[:-shape[0]])
        logp_grad, f_tilde = divergence_eval_wrapper(v, time_steps)
        sample_grad = f_tilde * (1 / (np.fmax(v, eps)) + 1 / np.fmax(1 - v, eps))

        return np.concatenate([sample_grad, logp_grad], axis=0)

    init = np.concatenate([logodds(v.detach().cpu().numpy().reshape((-1,))), np.zeros((shape[0],))], axis=0)
    res = integrate.solve_ivp(ode_func, (min_time, max_time), init, rtol=1e-4, atol=1e-4, method='RK23')
    zp = res.y[:, -1]
    z = torch.Tensor(inverse_logodds(zp[:-shape[0]])).to(device).reshape(shape)

    delta_logp = torch.Tensor(zp[-shape[0]:]).to(device).reshape(shape[0])
    prior_logp = prior_likelihood(z, alpha, beta)
    sb_delta_logp = sb.log_abs_det_jacobian(v.to(device))
    if v.ndim - 1 > 1:
        sb_delta_logp = sb_delta_logp.sum(dim=tuple(range(1, v.ndim - 1)))

    print(f"Number of function evaluations: {res.nfev}")
    if verbose:
        print("prior_logp", prior_logp)
        print("delta_logp", delta_logp)
        print("sb_delta_logp", sb_delta_logp)

    return z, prior_logp + delta_logp + sb_delta_logp



