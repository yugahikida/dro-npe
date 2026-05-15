from collections.abc import Sequence
from typing import Optional
import bayesflow
import keras
import torch.nn.functional as F

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import concatenate_valid

from bayesflow.approximators.continuous_approximator import ContinuousApproximator
from src.utils.calnpe_helper import STEFunctionRanksq, get_coverage
import torch

class ContinuousApproximatorCAL(ContinuousApproximator):
    """
    Implementation of calibrated posterior paper (https://arxiv.org/abs/2310.13402)
    following https://github.com/DMML-Geneva/calibrated-posterior/blob/main/src/calibration/npe/snpe.py

    Parameters
    ----------
    adapter : bayesflow.adapters.Adapter
        Adapter for data processing. You can use :py:meth:`build_adapter`
        to create it.
    inference_network : InferenceNetwork
        The inference network used for posterior or likelihood approximation.
    summary_network : SummaryNetwork, optional
        The summary network used for data summarization (default is None).
    standardize : str | Sequence[str] | None
        The variables to standardize before passing to the networks. Can be either
        "all" or any subset of ["inference_variables", "summary_variables", "inference_conditions"].
        (default is "all").
    calibration: bool = False
        Whether to use calibration loss or conservation loss for training. 
        Calibration loss penalises the deviation from uniformality in both directions, 
        whereas the conservation loss penalizes only positive deviations, thereby specifically discouraging overconfidence.
    gamma: float
        Regularization parameter for the calibration loss.
    L: int
        Number of samples to draw from the proposal (prior) distribution to calculate the calibration loss.
    prior_sampler: torch.distributions.Distribution, optional
        Torch distribution used to sample prior parameters. Need to be passed when training.
    **kwargs : dict, optional
        Additional arguments passed to the :py:class:`bayesflow.approximators.Approximator` class.
    """

    CONDITION_KEYS = ["summary_variables", "inference_conditions"]

    def __init__(
        self,
        *,
        adapter: Adapter,
        inference_network: InferenceNetwork,
        summary_network: SummaryNetwork = None,
        standardize: str | Sequence[str] | None = "all",
        calibration: bool = False,
        gamma: float = 5.0,
        L: int = 16,
        prior_sampler: Optional[torch.distributions.Distribution] = None,
        **kwargs,
    ):
        super().__init__(
            adapter=adapter,
            inference_network=inference_network,
            summary_network=summary_network,
            standardize=standardize,
            **kwargs,
        )
        self.adapter = adapter
        self.inference_network = inference_network
        self.summary_network = summary_network
        self.calibration = calibration
        self.gamma = gamma
        self.L = L
        self.prior_sampler = prior_sampler

    def get_log_density_for_ranks(self, inference_variables, inference_conditions, stage):
        # Here inference_variables and inference_conditions must be already preprocessed in compute_metric (standardisation etc)

        batch_size = inference_variables.shape[0]

        inference_conditions_repeated = (
            inference_conditions.unsqueeze(1)
            .expand(-1, self.L, -1)
            .reshape(batch_size * self.L, -1)
        ) # [batch_size * L, d_x]
    
        #  Obtain L * batch_size samples from the prior distribution and process them
        theta_prior_samples_raw = self.prior_sampler.sample((self.L * batch_size,))  # CPU
        theta_prior_samples_on_device = theta_prior_samples_raw.to(inference_variables.device)
        theta_prior_samples = self._prepare_inference_variables(theta_prior_samples_on_device, stage=stage) # [batch_size * L, d_theta]

        # Compute log density of the true theta and the prior samples over batch_size different x.
        all_theta = torch.cat([inference_variables, theta_prior_samples], dim=0) # [batch_size * (1 + L), d_theta]
        all_conditions = torch.cat([inference_conditions, inference_conditions_repeated], dim=0) # [batch_size * (1 + L), d_x]
        all_log_density = self.inference_network.log_prob(all_theta, conditions=all_conditions) # [batch_size * (1 + L)]

        log_density_true = all_log_density[:batch_size].unsqueeze(1)  # [batch_size, 1]
        log_density_prior = all_log_density[batch_size:].reshape(batch_size, self.L)  # [batch_size, L]

        log_density = torch.cat((log_density_true, log_density_prior), dim=1) # [batch_size, 1 + L]
        return log_density, theta_prior_samples_raw.reshape(batch_size, self.L, -1)

    def get_ranks(self, inference_variables, inference_conditions, stage):
        log_density, theta_prior_samples_raw = self.get_log_density_for_ranks(inference_variables, inference_conditions, stage)
        density = torch.exp(log_density)
        log_prior_density = self.prior_sampler.log_prob(theta_prior_samples_raw).to(log_density.device) # [batch_size, L] or [batch_size, L, d_theta]
        
        if log_prior_density.dim() > 2:
            log_prior_density = log_prior_density.sum(dim=-1)
            
        I_theta = torch.exp(log_prior_density) # [batch_size, L]
        importance_weights = density[:, 1:] / I_theta # [batch_size, L]
        ranks = (importance_weights * STEFunctionRanksq.apply(density[:, 0].unsqueeze(1) - density[:, 1:])).sum(dim=1) / importance_weights.sum(dim=1) # [batch_size, L] -> [batch_size]
        # original code: q[:, 1:] * STEFunctionRanksq.apply(q[:, 0].unsqueeze(1) - q[:, 1:])).sum(dim=1) / q[:, 1:].sum(dim=1) (from original code)

        return ranks

    def compute_calibration_error(self, inference_variables, inference_conditions, sample_weight, stage):
        ranks = self.get_ranks(inference_variables, inference_conditions, stage)
        coverage, expected = get_coverage(ranks)
        if self.calibration:
            calibration_error = (expected - coverage).pow(2).mean()
        else:
            calibration_error = torch.clamp(expected - coverage, min=0).pow(2).mean() # conservativeness loss (relu has bug on mps apearently so avoud using it)
        return calibration_error

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        if keras.config.backend() != "torch":
            raise ValueError(
                "Training of ContinuousApproximatorCAL requires the Torch backend. " # if you put it __init__ then it will raise error when just importing class for evaluation
                f"Current backend is '{keras.config.backend()}'. "
                "Set KERAS_BACKEND=torch before importing keras."
            )
    
        summary_metrics, summary_outputs = self._compute_summary_metrics(summary_variables, stage=stage)

        if "inference_conditions" in self.standardize:
            inference_conditions = self.standardize_layers["inference_conditions"](inference_conditions, stage=stage)

        inference_conditions = concatenate_valid((inference_conditions, summary_outputs), axis=-1)
        inference_variables = self._prepare_inference_variables(inference_variables, stage=stage)

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=inference_conditions, sample_weight=sample_weight, stage=stage
        )

        if self.gamma > 0:
            regularization_term = self.gamma * self.compute_calibration_error(inference_variables, inference_conditions, sample_weight=sample_weight, stage=stage)

        else:
            regularization_term = torch.tensor(0.0, device=inference_variables.device)

        inference_metrics["regularization"] = regularization_term

        if "loss" in summary_metrics:
            loss = inference_metrics["loss"] + summary_metrics["loss"] + inference_metrics["regularization"]
        else:
            loss = inference_metrics["loss"] + inference_metrics["regularization"]

        inference_metrics = {f"{key}/inference_{key}": value for key, value in inference_metrics.items()}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = {"loss": loss} | inference_metrics | summary_metrics
        return metrics