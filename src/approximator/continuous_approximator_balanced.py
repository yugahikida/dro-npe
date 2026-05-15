from collections.abc import Sequence
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
import bayesflow
import keras
import distrax

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import concatenate_valid

from bayesflow.approximators.continuous_approximator import ContinuousApproximator
from typing import Optional

class ContinuousApproximatorBalanced(ContinuousApproximator):
    """
    Implementation of balanced NPE (https://github.com/ADelau/balancing_sbi/blob/8047c1a09ba278d3b0ada075bcbcdb67a335e25f/src/balancing_sbi/models/bnpe.py)

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
    lmbda: float
        Regularization parameter for the regularization term.
    prior_sampler: distrax.Distribution
        Prior sampler used to compute the balancing term. (use log prob)
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
        lmbda: float = 100,
        prior_sampler: Optional[distrax.Distribution] = None,
        **kwargs,
    ):
        if keras.config.backend() != "jax":
            raise ValueError(
                "ContinuousApproximatorBalanced requires the JAX backend. "
                f"Current backend is '{keras.config.backend()}'. "
                "Set KERAS_BACKEND=jax before importing keras."
            )
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
        self.lmbda = lmbda
        self.prior_sampler = prior_sampler

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:

        summary_metrics, summary_outputs = self._compute_summary_metrics(summary_variables, stage=stage)

        if summary_outputs is not None:
            raise ValueError("Summary variables are not supported for balanced NPE yet")

        if "inference_conditions" in self.standardize:
            inference_conditions = self.standardize_layers["inference_conditions"](inference_conditions, stage=stage)

        inference_conditions = concatenate_valid((inference_conditions, summary_outputs), axis=-1)
        inference_variables_raw = inference_variables
        inference_variables = self._prepare_inference_variables(inference_variables_raw, stage=stage)

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=inference_conditions, sample_weight=sample_weight, stage=stage
        )

        if self.lmbda > 0:
            theta_prime_raw = jnp.roll(inference_variables_raw, 1, 0) # torch.roll(theta, 1, dims=0) (original code)
            theta_prime = self._prepare_inference_variables(theta_prime_raw, stage=stage)
            theta_concat = jnp.concatenate((theta_prime, inference_variables), axis = 0) # [2 * batch_size, d_theta]
            inference_conditions_rep = inference_conditions.repeat(2, axis = 0) # [2 * batch_size, d_x]
            log_density_concat = self.inference_network.log_prob(theta_concat, conditions=inference_conditions_rep) # [2 * batch_size]
            log_density_prime = log_density_concat[:len(theta_prime)] # [batch_size]
            log_density = log_density_concat[len(theta_prime):] # [batch_size]

            log_density_prior = self.prior_sampler.log_prob(inference_variables_raw) # [batch_size, d_theta] or [batch_size]
            log_density_prior_prime = self.prior_sampler.log_prob(theta_prime_raw)

            if log_density_prior.ndim > 1: # [batch_size, d_theta] -> [batch_size]
                log_density_prior = log_density_prior.sum(axis=-1)
                log_density_prior_prime = log_density_prior_prime.sum(axis=-1)

            balancing_term = jnp.square((sigmoid(log_density - log_density_prior) + sigmoid(log_density_prime - log_density_prior_prime) - 1.0).mean())
            #  lb = (torch.sigmoid(log_p - self.prior.log_prob(theta)) + torch.sigmoid(log_p_prime - self.prior.log_prob(theta_prime)) - 1).mean().square() (from original code)

            regularization_term = self.lmbda * balancing_term

        else:
            regularization_term = keras.ops.array(0.0)

        inference_metrics["regularization"] = regularization_term

        if "loss" in summary_metrics:
            raise ValueError("Summary metrics are not supported for balanced NPE yet")
        else:
            loss = inference_metrics["loss"] + inference_metrics["regularization"]

        inference_metrics = {f"{key}/inference_{key}": value for key, value in inference_metrics.items()}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = {"loss": loss} | inference_metrics | summary_metrics
        return metrics
   