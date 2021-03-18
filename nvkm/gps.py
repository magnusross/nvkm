#%%
from functools import partial

import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy as jsp
import matplotlib.pyplot as plt
from jax import jit, vmap
from stheno import Normal
from varz import Vars

from .settings import JITTER
from .utils import l2p, map2matrix, eq_kernel


class EQApproxGP:
    def __init__(
        self,
        z: jnp.DeviceArray = None,
        v: jnp.DeviceArray = None,
        N_basis: int = 500,
        D: int = 1,
        ls: float = 1.0,
        amp: float = 1.0,
        noise: float = 0.0,
    ):

        self.z = z
        self.v = v
        self.N_basis = N_basis
        self.D = D

        self.ls = ls
        self.amp = amp
        self.noise = noise

        self.Kvv = None
        self.LKvv = None

        if self.z == None:
            pass

        else:
            try:
                assert self.z.shape[1] == self.D

            except IndexError:
                self.z = self.z.reshape(-1, 1)
                assert self.D == 1

            except AssertionError:
                raise ValueError(
                    "Dimension of inducing points does not match dimension of GP."
                )

            self.Kvv, self.LKvv = self.compute_covariances(amp, ls)

    @partial(jit, static_argnums=(0,))
    def compute_covariances(self, amp, ls):
        Kvv = map2matrix(self.kernel, self.z, self.z, amp, ls) + (
            self.noise + JITTER
        ) * jnp.eye(self.z.shape[0])
        LKvv = jnp.linalg.cholesky(Kvv)
        return Kvv, LKvv

    @partial(jit, static_argnums=(0,))
    def compute_Phi(self, thetas, betas):
        return vmap(lambda zi: self.phi(zi, thetas, betas))(self.z)

    @partial(jit, static_argnums=(0,))
    def kernel(self, t, tp, amp, ls):
        return eq_kernel(t, tp, amp, ls)

    @partial(jit, static_argnums=(0, 2))
    def sample_thetas(self, key, shape, ls):
        # FT of isotropic gaussain is inverser varience
        return jrnd.normal(key, shape) / ls

    @partial(jit, static_argnums=(0, 2))
    def sample_betas(self, key, shape):
        return jrnd.uniform(key, shape, maxval=2 * jnp.pi)

    @partial(jit, static_argnums=(0, 2))
    def sample_ws(self, key, shape):
        return jrnd.normal(key, shape)

    @partial(jit, static_argnums=(0,))
    def phi(self, t, theta, beta):
        return jnp.sqrt(2 / self.N_basis) * jnp.cos(jnp.dot(theta, t) + beta)

    # @partial(jit, static_argnums=(0,))
    def compute_q(self, thetas, betas, ws):
        Phi = self.compute_Phi(thetas, betas)
        b = self.v - Phi @ ws
        print(b.shape)
        return jsp.linalg.cho_solve((self.LKvv, True), b)

    def sample(self, t, Ns=100, key=jrnd.PRNGKey(1)):

        try:
            assert t.shape[1] == self.D

        except IndexError:
            t = t.reshape(-1, 1)
            assert self.D == 1

        except AssertionError:
            raise ValueError(
                "Dimension of inducing points does not match dimension of GP."
            )
        # sample random parameters
        skey = jrnd.split(key, 3)
        thetas = self.sample_thetas(skey[0], (Ns, self.N_basis, self.D), self.ls)
        betas = self.sample_betas(skey[1], (Ns, self.N_basis))
        ws = self.sample_ws(skey[2], (Ns, self.N_basis))

        # fourier basis part
        samps = vmap(
            lambda ti: vmap(lambda thi, bi, wi: jnp.dot(wi, self.phi(ti, thi, bi)))(
                thetas, betas, ws
            )
        )(t)

        # canonical basis part
        if self.z == None:
            pass
        else:
            qs = vmap(lambda thi, bi, wi: self.compute_q(thi, bi, wi))(
                thetas, betas, ws
            )  # Ns x Nz
            kv = map2matrix(self.kernel, t, self.z, self.amp, self.ls)  # Nt x Nz
            # print(qs.shape, kv.shape)
            kv = jnp.einsum("ij, kj", qs, kv)  # Nt x Ns

            samps += kv.T

        return samps


# %%
# key = jrnd.PRNGKey(10)
# X = jnp.arange(-1, 1, 0.1)
# Y = jnp.arange(-1, 1, 0.1)
# xx, yy = jnp.meshgrid(X, Y)
# tt = jnp.array([xx.flatten(), yy.flatten()]).T

# # %%
# zt = 2.0 * jrnd.uniform(key, (10, 2)) - 1.0
# vt = 2 * jnp.cos(jnp.dot(jnp.ones((2,)), zt.T)) + jnp.dot(jnp.ones((2,)), zt.T)

# test = EQApproxGP(D=2, v=vt, z=zt, ls=0.3)
# samps = test.sample(tt, Ns=2)

# fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"projection": "3d"})
# # for i in range(len(samps[0])):
# surf = ax.plot_surface(xx, yy, samps[:, 0].reshape(len(X), len(Y)), alpha=0.8)
# surf = ax.plot_surface(xx, yy, samps[:, 1].reshape(len(X), len(Y)), alpha=0.8)
# ax.scatter(zt[:, 0], zt[:, 1], vt, c="red", marker="x")
# plt.show()


# # %%
# t = jnp.linspace(-10, 10, 500)

# fig = plt.figure(figsize=(20, 10))
# z = jnp.linspace(-3, 3, 10)
# v = 2 * jnp.cos(z) + z
# testd = EQApproxGP(z=z, v=v, ls=0.5, noise=0.0, N_basis=1000)
# sampsd = testd.sample(t, Ns=1)
# plt.plot(t, sampsd)
# plt.show()
# %%

# %%
