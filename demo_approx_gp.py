import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

from nvkm.gps import EQApproxGP
import nvkm.utils as utils
from nvkm.settings import JITTER

t = jnp.linspace(-10, 10, 500)

z = jnp.linspace(-3, 3, 10)
v = 2 * jnp.cos(z) + z - jnp.sin(z) ** 2

N_samples = 30

gp1D = EQApproxGP(z=z, v=v, amp=1.0, ls=0.5, noise=0.01, N_basis=1000)
samps = gp1D.sample(t, Ns=N_samples)

exact_m, exact_cov = utils.exact_gp_posterior(
    utils.eq_kernel, t, z, v, 1.0, 0.5, noise=0.01
)
exact_samps = jrnd.multivariate_normal(
    jrnd.PRNGKey(1000),
    exact_m,
    exact_cov + JITTER * jnp.eye(len(t)),
    shape=(N_samples,),
).T

fig, axs = plt.subplots(3, 1, figsize=(10, 10))

axs[0].scatter(z, v, marker="x", c="green", label="Data")
axs[0].plot(t, jnp.mean(samps, axis=1), c="red", ls=":", alpha=1.0, label="Approx.")
axs[0].plot(t, exact_m, c="green", ls=":", alpha=1.0, label="Exact")
axs[0].title.set_text("Means")
axs[0].legend()

axs[1].scatter(z, v, marker="x", c="green", label="Data")
axs[1].plot(t, exact_samps, c="blue", alpha=0.1)
axs[1].title.set_text("Exact Samples")

axs[2].scatter(z, v, marker="x", c="green", label="Data")
axs[2].plot(t, samps, c="blue", alpha=0.1)
axs[2].title.set_text("Approx. Samples")
plt.show()

