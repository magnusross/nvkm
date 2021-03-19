import pytest
from nvkm import models
from nvkm import utils
from nvkm.settings import JITTER
import jax.numpy as jnp
import jax.random as jrnd


@pytest.fixture
def set_a():
    a = 1
    return a


class TestEQApproxGP:
    def test_1D_dist(self):
        t = jnp.linspace(-10, 10, 100)
        # inducing data
        z = jnp.linspace(-3, 3, 10)
        v = 2 * jnp.cos(z) + z - jnp.sin(z) ** 2
        N_samples = 1000
        # Make approx GP in 1D case
        gp1D = models.EQApproxGP(
            z=z, v=v, D=1, amp=1.0, ls=0.5, noise=0.01, N_basis=1000
        )
        samps = gp1D.sample(t, Ns=N_samples, key=jrnd.PRNGKey(999))

        exact_m, exact_cov = utils.exact_gp_posterior(
            utils.eq_kernel, t, z, v, 1.0, 0.5, noise=0.01
        )
        # Sample exactly
        exact_samps = jrnd.multivariate_normal(
            jrnd.PRNGKey(1000),
            exact_m,
            exact_cov + JITTER * jnp.eye(len(t)),
            shape=(N_samples,),
        ).T
        mean_err = utils.RMSE(jnp.mean(samps, axis=1), exact_m)
        var_error = utils.RMSE(jnp.var(samps, axis=1), jnp.diag(exact_cov))
        assert mean_err < 0.05 and var_error < 0.05

    def test_5D_diag(self):
        N = 5
        N_samples = 1000
        key = jrnd.PRNGKey(1000)
        tdiag = jnp.linspace(-1, 1, 30)
        ttdiag = jnp.vstack((tdiag for i in range(N))).T
        zt = 2.0 * jrnd.uniform(key, (30, N)) - 1.0
        vt = 2 * jnp.cos(jnp.dot(jnp.ones((N,)), zt.T)) + jnp.dot(jnp.ones((N,)), zt.T)

        gp2D = models.EQApproxGP(
            D=N, v=vt, z=zt, amp=1.0, ls=0.5, noise=0.01, N_basis=1000
        )
        samps_diag = gp2D.sample(ttdiag, Ns=N_samples)

        exact_m_diag, exact_cov_diag = utils.exact_gp_posterior(
            utils.eq_kernel, ttdiag, zt, vt, 1.0, 0.5, noise=0.01
        )
        # Sample exactly
        exact_samps_diag = jrnd.multivariate_normal(
            jrnd.PRNGKey(1000),
            exact_m_diag,
            exact_cov_diag + JITTER * jnp.eye(len(ttdiag)),
            shape=(N_samples,),
        ).T

        mean_err = utils.RMSE(jnp.mean(samps_diag, axis=1), exact_m_diag)
        var_error = utils.RMSE(jnp.var(samps_diag, axis=1), jnp.diag(exact_cov_diag))
        assert mean_err < 0.05 and var_error < 0.05
