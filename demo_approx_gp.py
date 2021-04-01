#%%
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

from nvkm.models import EQApproxGP
import nvkm.utils as utils
from nvkm.settings import JITTER


#%%
# test inputs
t = jnp.linspace(-10, 10, 500)

# inducing data
z = jnp.linspace(-3, 3, 10)
v = 2 * jnp.cos(z) + z - jnp.sin(z) ** 2

N_samples = 100

# Make approx GP in 1D case
gp1D = EQApproxGP(z=z, v=v, amp=0.01, ls=0.5, noise=0.01, N_basis=1000)
samps = gp1D.sample(t, Ns=N_samples)

# Get exact mean and cov
exact_m, exact_cov = utils.exact_gp_posterior(
    utils.eq_kernel, t, z, v, 0.01, 0.5, noise=0.01
)
# Sample exactly
exact_samps = jrnd.multivariate_normal(
    jrnd.PRNGKey(1000),
    exact_m,
    exact_cov + JITTER * jnp.eye(len(t)),
    shape=(N_samples,),
).T


fig, axs = plt.subplots(3, 1, figsize=(10, 10))
# plot means
axs[0].scatter(z, v, marker="x", c="green", label="Data")
axs[0].plot(t, jnp.mean(samps, axis=1), c="red", ls=":", alpha=1.0, label="Approx.")
axs[0].plot(t, exact_m, c="green", ls=":", alpha=1.0, label="Exact")
axs[0].title.set_text("Means")
axs[0].legend()
# Plot exact sample
axs[1].scatter(z, v, marker="x", c="green", label="Data")
axs[1].plot(t, exact_samps, c="blue", alpha=0.1)
axs[1].title.set_text("Exact Samples")
# Plot approximate samples
axs[2].scatter(z, v, marker="x", c="green", label="Data")
axs[2].plot(t, samps, c="blue", alpha=0.1)
axs[2].title.set_text("Approx. Samples")
plt.savefig("basegp.png")
plt.show()
#%%
print(utils.RMSE(jnp.mean(samps, axis=1), exact_m))
print(utils.RMSE(jnp.var(samps, axis=1), jnp.diag(exact_cov)))
#%%
# now check 2D means with surface plot
key = jrnd.PRNGKey(10)
N_grid = 25
x = jnp.linspace(-1, 1, N_grid)
y = jnp.linspace(-1, 1, N_grid)
xx, yy = jnp.meshgrid(x, y)
tt = jnp.array([xx.flatten(), yy.flatten()]).T

# %%
zt = 2.0 * jrnd.uniform(key, (30, 2)) - 1.0
vt = 2 * jnp.cos(jnp.dot(jnp.ones((2,)), zt.T)) + jnp.dot(jnp.ones((2,)), zt.T)

exact_mt, exact_covt = utils.exact_gp_posterior(
    utils.eq_kernel, tt, zt, vt, 1.0, 0.5, noise=0.01
)

gp2D = EQApproxGP(D=2, v=vt, z=zt, amp=1.0, ls=0.5, noise=0.01, N_basis=1000)
samps = gp2D.sample(tt, Ns=N_samples)

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})


ax.scatter(zt[:, 0], zt[:, 1], vt, s=100.0, c="red", marker="x")
surf = ax.plot_surface(
    xx, yy, jnp.mean(samps, axis=1).reshape(N_grid, N_grid), alpha=0.5
)
surf = ax.plot_surface(xx, yy, exact_mt.reshape(N_grid, N_grid), alpha=0.5)
plt.show()
# %%
# now let's check the main diagonal in N dimensions
#
def plot_ND_samples(N):
    tdiag = jnp.linspace(-1, 1, 30)
    ttdiag = jnp.vstack((tdiag for i in range(N))).T
    zt = 2.0 * jrnd.uniform(key, (30, N)) - 1.0
    vt = 2 * jnp.cos(jnp.dot(jnp.ones((N,)), zt.T)) + jnp.dot(jnp.ones((N,)), zt.T)

    gp2D = EQApproxGP(D=N, v=vt, z=zt, amp=1.0, ls=0.5, noise=0.01, N_basis=1000)
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

    fig, axs = plt.subplots(2, 1, figsize=(7, 7))

    axs[0].plot(tdiag, exact_samps_diag, c="blue", alpha=0.1)
    axs[0].title.set_text(f"Exact Samples {N}D, main diagonal")
    # Plot approximate samples

    axs[1].plot(tdiag, samps_diag, c="blue", alpha=0.1)
    axs[1].title.set_text(f"Approx. Samples {N}D, main diagonal")
    plt.show()

    print(utils.RMSE(jnp.mean(samps_diag, axis=1), exact_m_diag))
    print(utils.RMSE(jnp.var(samps_diag, axis=1), jnp.diag(exact_cov_diag)))


# %%
plot_ND_samples(2)
plot_ND_samples(4)
plot_ND_samples(10)
# %%
