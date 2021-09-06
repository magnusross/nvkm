from nvkm.utils import l2p, make_zg_grids
from nvkm.models import EQApproxGP, SepEQApproxGP

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif"],
    }
)

#%%
z, ls = make_zg_grids([1.0, 1.0], [10, 10])
z_sep, z_full = z
full_gp = EQApproxGP(z_full, None, N_basis=50, D=2, ls=ls[0])
sep_gp = SepEQApproxGP(z_sep, None, N_basis=50, D=2, ls=ls[1])
hom_gp = EQApproxGP(z_sep, None, N_basis=50, D=1, ls=ls[0])

alpha = 3
# %%
N = 50
t1 = jnp.linspace(-1.1, 1.1, N)
x = jnp.meshgrid(t1, t1)
t2 = jnp.vstack((x[0].flatten(), x[1].flatten())).T

normer = lambda t: jnp.exp(-alpha * jnp.sum(t ** 2, axis=1))
# %%
S = 4
fig, axs = plt.subplots(S, 3, figsize=(6, 1.5 * S))
axs = axs.T
for i in range(S):
    sm = (normer(t2) * full_gp.sample(t2, 1, key=jrnd.PRNGKey(i)).T).reshape(N, N)

    axs[0, i].imshow(
        (sm + sm.T) / 2,
        interpolation="gaussian",
        cmap="Reds",
    )
    axs[0, i].axis("off")

axs[0, 0].set_title("Full")

for i in range(S):
    sm = (normer(t2) * sep_gp.sample(t2, 1, key=jrnd.PRNGKey(i)).T).reshape(N, N)
    axs[1, i].imshow(
        (sm + sm.T) / 2,
        interpolation="gaussian",
        cmap="Reds",
    )
    axs[1, i].axis("off")

axs[1, 0].set_title("Separable")
for i in range(S):
    s1 = hom_gp.sample(t2[:, 0].reshape(-1, 1), 1, key=jrnd.PRNGKey(i))
    s2 = hom_gp.sample(t2[:, 1].reshape(-1, 1), 1, key=jrnd.PRNGKey(i))
    axs[2, i].imshow(
        (normer(t2) * s1.T * s2.T).reshape(N, N), interpolation="gaussian", cmap="Reds"
    )
    axs[2, i].axis("off")
axs[2, 0].set_title("Homog.")

plt.tight_layout()
fig.patch.set_visible(False)
plt.savefig("filters.pdf")
plt.show()
# %%

# %%

# %%
