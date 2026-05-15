import numpy as np
import jax.numpy as jnp
import distrax


class LogNormal:
    def __init__(self, loc, scale):
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)

    def log_prob(self, x):
        x = jnp.asarray(x)
        logx = jnp.log(x)
        logp = (
            -jnp.log(x)
            -jnp.log(self.scale)
            -0.5 * jnp.log(2.0 * jnp.pi)
            -0.5 * ((logx - self.loc) / self.scale) ** 2
        )
        return jnp.where(x > 0, logp, -jnp.inf)

    def prob(self, x):
        return jnp.exp(self.log_prob(x))

def get_prior_sampler(simulator_name):
    if simulator_name == "slcp":
        prior_sampler = distrax.Uniform(
            low=jnp.full((5,), -3.0),
            high=jnp.full((5,), 3.0),
        )
    elif simulator_name == "lv":
        prior_sampler = LogNormal(
            loc=jnp.array([-0.125, -3.0, -0.125, -3.0]),
            scale=jnp.full((4,), 0.5),
        )
    elif simulator_name == "ik":
        prior_sampler = distrax.Normal(
            loc=jnp.zeros(4),
            scale=jnp.array([0.25, 0.5, 0.5, 0.5]),
        )

    elif simulator_name == "tm":
        prior_sampler = distrax.Uniform(
            low=jnp.full((2,), -1.0),
            high=jnp.full((2,), 1.0),
        )

    elif simulator_name == "cosmo":
        prior_sampler = distrax.Uniform(
            low=jnp.array([0.1, 0.6]),
            high=jnp.array([0.5, 1.0]),
        )
    else:
        raise ValueError(f"Simulator {simulator_name} not supported for CAL yet")
    return prior_sampler