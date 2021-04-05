from jax.config import config


import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrnd
from jax import jit


from typing import Dict, Union, List
from functools import partial

config.update("jax_enable_x64", True)

VIPars = Dict[str, Union[jnp.DeviceArray, List[jnp.DeviceArray]]]


class VariationalDistribution:
    def __init__(self, init_pars: Union[VIPars, None]):
        self.q_pars = init_pars

    def initialize(self, data):
        pass

    def _KL(self, p_pars, q_pars):
        pass

    def _sample(self, q_pars, N_s, key):
        pass

    def KL(self, p_pars):
        return self._KL(p_pars, self.q_pars)

    def sample(self, key, N_s=10):
        return self._sample(self.q_pars, N_s, key)


class IndependentGaussians:
    def __init__(self):
        """[summary]

        Args:
            init_pars (Dict[str, jnp.DeviceArray]):
            should have fields "LCu", "mu",
        """
        # super().__init__(init_pars)

    def initialize(self, model, frac, key=jrnd.PRNGKey(110011)):
        skey = jrnd.split(key, 2)
        q_pars = {}
        q_pars["LC_gs"] = [arr * frac for arr in model.p_pars["LK_gs"]]
        q_pars["LC_u"] = frac * model.p_pars["LK_u"]
        q_pars["mu_gs"] = [
            gp.sample(gp.z, 1, key=skey[0]).flatten() for gp in model.g_gps
        ]
        q_pars["mu_u"] = model.u_gp._sample(
            model.u_gp.z, model.u_gp.v, model.u_gp.amp, 1, key=skey[1]
        ).flatten()
        return q_pars

    @partial(jit, static_argnums=(0,))
    def single_KL(self, LC, m, LK):
        C = LC @ LC.T
        mt = -0.5 * (
            jnp.dot(m.T, jsp.linalg.cho_solve((LK, True), m))
            + jnp.trace(jsp.linalg.cho_solve((LK, True), C))
        )
        st = jnp.sum(jnp.log(jnp.diag(LC) / jnp.diag(LK))) + 0.5 * LC.shape[0]
        return mt + st

    @partial(jit, static_argnums=(0,))
    def _KL(self, p_pars, q_pars):
        val = 0.0
        for i in range(len(q_pars["LC_gs"])):
            val += self.single_KL(
                q_pars["LC_gs"][i], q_pars["mu_gs"][i], p_pars["LK_gs"][i]
            )
        val += self.single_KL(q_pars["LC_u"], q_pars["mu_u"], p_pars["LK_u"])
        return val

    @partial(jit, static_argnums=(0, 2))
    def _sample(self, q_pars, N_s, key):
        D = len(q_pars["LC_gs"])
        keys = jrnd.split(key, D + 1)
        samps_dict = {"u": None, "gs": []}

        for i in range(D):
            LC_g = q_pars["LC_gs"][i]

            samps_dict["gs"].append(
                jrnd.multivariate_normal(
                    keys[i], q_pars["mu_gs"][i], LC_g @ LC_g.T, (N_s,)
                )
            )

        samps_dict["u"] = jrnd.multivariate_normal(
            keys[-1], q_pars["mu_u"], q_pars["LC_u"] @ q_pars["LC_u"].T, (N_s,)
        )

        return samps_dict


@jit
def gaussain_likelihood(y, samples, noise):
    """
    gaussian likelihood
    """
    Nt = samples.shape[1]
    Ns = samples.shape[0]
    C = -0.5 * Nt * jnp.log(2 * jnp.pi * noise ** 2)
    return C - (1 / Ns) * (1 / (2 * noise ** 2)) * jnp.sum((y - samples.T) ** 2)
