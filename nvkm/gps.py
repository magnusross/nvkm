from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrnd
from jax import jit, vmap

from varz import Vars
from stheno import Normal

from utils import map2matrix, l2p

# import settings

JITTER = 1e-5


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
        elif self.z.shape[1] != self.D:
            raise ValueError("Dimension of inducing points incorrect")
        else:
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
        return self.amp ** 2 * jnp.exp(-0.5 * jnp.sum((t - tp) ** 2) / self.ls ** 2)

    @partial(jit, static_argnums=(0, 2))
    def sample_thetas(self, key, shape, ls):
        return jrnd.normal(key, shape) / ls

    @partial(jit, static_argnums=(0, 2))
    def sample_betas(self, key, shape):
        return jrnd.uniform(key, shape, maxval=2 * jnp.pi)

    @partial(jit, static_argnums=(0, 2))
    def sample_ws(self, key, shape):
        return jrnd.normal(key, shape)

    @partial(jit, static_argnums=(0,))
    def phi(self, t, theta, beta):
        return jnp.sqrt(2 / self.N_basis) * jnp.cos(theta * t + beta)

    @partial(jit, static_argnums=(0,))
    def compute_q(self, thetas, betas, ws):
        Phi = self.compute_Phi(thetas, betas)
        b = self.v - Phi @ ws
        return jsp.linalg.cho_solve((self.LKvv, True), b)

    def sample(self, t, Ns=100, key=jrnd.PRNGKey(1)):
        # posterior part
        skey = jrnd.split(key, 3)
        thetas = self.sample_thetas(skey[0], (Ns, self.N_basis), self.ls)
        betas = self.sample_betas(skey[1], (Ns, self.N_basis))
        ws = self.sample_ws(skey[2], (Ns, self.N_basis))

        samps = vmap(
            lambda ti: vmap(lambda thi, bi, wi: jnp.dot(wi, self.phi(ti, thi, bi)))(
                thetas, betas, ws
            )
        )(t)

        if self.z == None:
            pass
        else:
            qs = vmap(lambda thi, bi, wi: self.compute_q(thi, bi, wi))(
                thetas, betas, ws
            )  # Ns x Nz
            kv = map2matrix(self.kernel, t, self.z, self.amp, self.ls)  # Nt x Nz
            kv = jnp.einsum("ij, kj", qs, kv)  # Nt x Ns

            samps += kv.T

        return samps


tt = ones((10, 2))


test = EQApproxGP()
t = jnp.linspace(-10, 10, 500)
samps = test.sample(t)

import matplotlib.pyplot as plt

plt.plot(t, samps)
plt.show()

fig = plt.figure(figsize=(20, 10))
z = jnp.linspace(-3, 3, 10)
v = 2 * jnp.cos(z) + z
testd = EQApproxGP(z=z, v=v, ls=0.5, noise=0.0, N_basis=1000)
sampsd = testd.sample(t, Ns=1)
plt.plot(t, sampsd)
plt.show()

