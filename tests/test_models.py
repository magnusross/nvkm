import pytest
from nvkm import models
from nvkm import utils
from nvkm import vi
from nvkm.settings import JITTER
import jax.numpy as jnp
import jax.random as jrnd


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

        mean_err = utils.RMSE(jnp.mean(samps_diag, axis=1), exact_m_diag)
        var_error = utils.RMSE(jnp.var(samps_diag, axis=1), jnp.diag(exact_cov_diag))
        assert mean_err < 0.05 and var_error < 0.05

    # def test_fast_covariance_recompute(self):
    #     z = jnp.linspace(-3, 3, 10)
    #     gp1a = models.EQApproxGP(
    #         z=z, v=None, D=1, amp=0.1, ls=0.5, noise=0.01, N_basis=1000
    #     )
    #     Kvva, LKvva = gp1a.fast_covariance_recompute(1.5)
    #     gp1b = models.EQApproxGP(
    #         z=z, v=None, D=1, amp=1.5, ls=0.5, noise=0.01, N_basis=1000
    #     )
    #     Kvvb, LKvvb = gp1b.Kvv, gp1b.LKvv
    #     print(LKvva, LKvvb)
    #     print(Kvva, Kvvb)
    #     assert jnp.all(jnp.isclose(LKvva, LKvvb)) and jnp.all(jnp.isclose(Kvva, Kvvb))


@pytest.fixture
def set_var_nvkm():
    keys = jrnd.split(jrnd.PRNGKey(5), 10)

    t1 = jnp.linspace(-2.0, 2, 2).reshape(-1, 1)
    t2 = 2 * jrnd.uniform(keys[0], shape=(2, 2)) - 1.0
    t3 = 2 * jrnd.uniform(keys[0], shape=(2, 3)) - 1.0

    x = jnp.linspace(-5, 5, 1)
    data = (x, jnp.sin(x))
    q_pars_init3 = {
        "LC_gs": [jnp.eye(2) for i in range(3)],
        "mu_gs": [
            jnp.sin(t1).flatten(),
            jnp.sin(t2[:, 0] ** 2),
            jnp.sin(t3[:, 0] ** 2),
        ],
        "LC_u": jnp.eye(2),
        "mu_u": jrnd.normal(keys[1], shape=(2,)),
    }
    return models.VariationalNVKM(
        [t1, t2, t3],
        jnp.linspace(-5, 5, 2).reshape(-1, 1),
        data,
        vi.IndependentGaussians,
        q_pars_init=q_pars_init3,
        lsgs=[1.0, 2.0, 1.0],
        ampgs=[1.0, 1.0, 1.0],
        noise=0.01,
        C=3,
    )


def set_nvkm():
    keys = jrnd.split(jrnd.PRNGKey(5), 10)
    t = jnp.linspace(-20, 20, 200)

    t1 = jnp.linspace(-2.0, 2, 10).reshape(-1, 1)
    t2 = 2 * jrnd.uniform(keys[0], shape=(5, 2)) - 1.0
    t3 = 2 * jrnd.uniform(keys[0], shape=(5, 3)) - 1.0
    return models.NVKM(
        zu=jnp.linspace(-10, 10, 20).reshape(-1, 1),
        vu=jrnd.normal(keys[1], shape=(20,)),
        zgs=[t1, t2, t3],
        vgs=[jnp.sin(t1).flatten(), jnp.sin(t2[:, 0] ** 2), jnp.sin(t3[:, 0] ** 2)],
        lsgs=[1.0, 2.0, 1.0],
        ampgs=[1.0, 1.0, 1.0],
        C=3,
    )


class TestVarNVKM:
    def test_fit(self, set_var_nvkm):
        model = set_var_nvkm
        b1 = model.compute_bound(3)
        model.fit(30, 1e-3, 1, 3)
        b2 = model.compute_bound(3)
        assert b1 > b2
