from collections.abc import Sequence
import jax
import jax.numpy as jnp
import bayesflow
import keras

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import concatenate_valid

from bayesflow.approximators.continuous_approximator import ContinuousApproximator

class ContinuousApproximatorDROUB(ContinuousApproximator):
    """
    Defines a workflow for performing fast posterior or likelihood inference.
    The distribution is approximated with an inference network and an optional summary network.

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
        epsilon: float = 0.0,
        **kwargs,
    ):
        if keras.config.backend() != "jax":
            raise ValueError(
                "ContinuousApproximatorDROUB requires the JAX backend. "
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
        self.epsilon = epsilon

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
            raise ValueError("Summary variables are not supported for dro_ub yet")

        if "inference_conditions" in self.standardize:
            inference_conditions = self.standardize_layers["inference_conditions"](inference_conditions, stage=stage)

        inference_conditions = concatenate_valid((inference_conditions, summary_outputs), axis=-1)
        inference_variables = self._prepare_inference_variables(inference_variables, stage=stage)

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=inference_conditions, sample_weight=sample_weight, stage=stage
        )

        if self.epsilon > 0:
            # TDOO: make the computation more efficient
            # define loss function
            def loss_fn(inference_variables, inference_conditions):
                """
                Compute  Σ_i - log q(theta_i | x_i)
                """
                out = self.inference_network.compute_metrics(
                    inference_variables,
                    conditions=inference_conditions,
                    sample_weight=sample_weight,
                    stage=stage,
                )

                batch_size = inference_variables.shape[0]
                return out["loss"] * batch_size # we need sum of loss across batches (instead of mean) to compute gradient w.r.t x and theta (otherwise gradient will be scaled by 1 / batch size)

            # calculate gradient of loss with respect to inference_variables (theta) and inference_conditions (x)
            grads = jnp.concatenate(jax.grad(loss_fn, argnums=(0, 1))(inference_variables, inference_conditions), axis = -1)

            # obtain regularisation term
            mean_grad_squared = keras.ops.mean(keras.ops.linalg.norm(grads, ord = 2, axis = -1) ** 2)
            regularization_term = self.epsilon * keras.ops.sqrt(mean_grad_squared)

        else:
            regularization_term = keras.ops.array(0.0)

        inference_metrics["regularization"] = regularization_term

        if "loss" in summary_metrics:
            raise ValueError("Summary metrics are not supported for dro_ub yet")
        else:
            loss = inference_metrics["loss"] + inference_metrics["regularization"]

        inference_metrics = {f"{key}/inference_{key}": value for key, value in inference_metrics.items()}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = {"loss": loss} | inference_metrics | summary_metrics
        return metrics
   