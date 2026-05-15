import torch
import torchsort
from torch.nn import functional as F
import numpy as np

# code taken from https://github.com/DMML-Geneva/calibrated-posterior/blob/main/src/calibration/npe/snpe.py

class STEFunctionRanksq(torch.autograd.Function):
    """
    Straight-through estimator for a hard indicator.

    Forward:
        returns 1 if input > 0, else 0

    Backward:
        passes a clipped gradient
    """

    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class STEFunctionRankslogq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.where(input > 0, 0, float("-inf"))

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)



def get_coverage(ranks):
    # Source: https://github.com/montefiore-ai/balanced-nre/blob/main/demo.ipynb
    # As a sample at a given rank belongs to the credible regions at levels 1-rank and below,
    # the coverage at level 1-alpha is the proportion of samples with ranks alpha and above.
    device = ranks.device
    ranks = ranks[~ranks.isnan()]

    # torchsort does not support MPS; compute on CPU and move back.
    if device.type == "mps":
        alpha = torchsort.soft_sort(ranks.to("cpu").unsqueeze(0)).squeeze().to(device)
    else:
        alpha = torchsort.soft_sort(ranks.unsqueeze(0)).squeeze()
    return (
        torch.linspace(0.0, 1.0, len(alpha) + 2, device=device)[1:-1],
        1 - torch.flip(alpha, dims=(0,)),
    )


def get_prior_sampler(simulator_name):
    if simulator_name == "slcp":
        lower_bound = torch.full((5,), -3.0, dtype=torch.float32)
        upper_bound = torch.full((5,), 3.0, dtype=torch.float32)
        prior_sampler = torch.distributions.Uniform(lower_bound, upper_bound)

    elif simulator_name == "lv":
        prior_sampler = torch.distributions.LogNormal(loc = torch.tensor([-0.125, -3, -0.125, -3]), scale = torch.tensor(0.5))

    elif simulator_name == "ik":
        prior_sampler = torch.distributions.Normal(
            loc=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            scale=torch.tensor([0.25, 0.5, 0.5, 0.5]),
        )

    elif simulator_name == "tm":
        prior_sampler = torch.distributions.Uniform(
            low=torch.full((2,), -1.0),
            high=torch.full((2,), 1.0),
        )

    elif simulator_name == "cosmo":
        prior_sampler = torch.distributions.Uniform(torch.tensor([0.1, 0.6]), torch.tensor([0.5, 1.0]))
    
    else:
        raise ValueError(f"Simulator {simulator_name} not supported for CAL yet")
    return prior_sampler