#%%
import argparse
from nvkm.utils import generate_C2_volterra_data
from nvkm.models import MOVarNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

#%%
Nvu = 20
Nvg = 25
O = 3

t1 = jnp.linspace(1, -1, Nvg).reshape(-1, 1)
tf = jnp.linspace(1, -1, int(jnp.sqrt(Nvg)))
tm2 = jnp.meshgrid(tf, tf)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T
t3 = 2 * jrnd.uniform(jrnd.PRNGKey(101), (Nvg, 3)) - 1

zgs = [[t1, t2], [t1, t2, t3], [t1, t2]]

var_model2 = MOVarNVKM(
    zgs,
    jnp.linspace(-17, 17, Nvu).reshape(-1, 1),
    None,
    q_pars_init=None,
    q_initializer_pars=0.001,
    lsgs=[[1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0]],
    ampgs=[[1.0, 0.8], [1.0, 0.3, 0.5], [0.1, 2.0]],
    noise=[0.1, 0.1, 0.1],
    alpha=[1.0, 1.0, 1.0],
    lsu=1.5,
    ampu=1.0,
    N_basis=50,
)

Nt = 150
t = jnp.linspace(-17, 17, Nt)
ts = [t] * O
#%%
# a = var_model2.q_of_v.sample(var_model2.q_pars, 1, jrnd.PRNGKey(1))


# %%
x = jnp.linspace(-15, 15, 50)
xs = [x] * O
ys = [i.flatten() for i in var_model2.sample(xs, 1)]
var_model2.data = (xs, ys)
# %%
var_model2.fit(500, 1e-3, 10, 5)
# %%
N_s = 10
tf = jnp.linspace(-3, 3, 100)

var_model2.plot_samples(t, ts, N_s)
#%%
var_model2.plot_filters(tf, N_s)
# %%

"""
fig, axs = plt.subplots(O + 1, 1, figsize=(10, 3.5 * (1 + O)))
for i in range(O):
    axs[i].set_ylabel(f"$y_{i+1}$")
    axs[i].set_xlabel("$t$")
    axs[i].plot(ts[i], samps[i], c="green", alpha=0.5)
    axs[i].scatter(var_model2.data[0][i], var_model2.data[1][i])

u_samps = var_model2.sample_u_gp(jnp.linspace(-17, 17, Nt), N_s)
axs[-1].set_ylabel(f"$u$")
axs[-1].set_xlabel("$t$")
axs[-1].scatter(var_model2.zu, var_model2.q_pars["mu_u"], c="blue", alpha=0.5)
axs[-1].plot(, u_samps, c="blue", alpha=0.5)
plt.savefig('samps.png', dpi=500)
plt.show()
# %%

tfs = [
    [jnp.vstack((tf for j in range(gp.D))).T for gp in var_model2.g_gps[i]]
    for i in range(var_model2.O)
]
g_samps = var_model2.sample_diag_g_gps(tfs, 10)
# %%
fig, axs = plt.subplots(
    max(var_model2.C), var_model2.O, figsize=(8 * var_model2.O, 5 * max(var_model2.C))
)
for i in range(var_model2.O):
    for j in range(var_model2.C[i]):
        y = g_samps[i][j].T * jnp.exp(-var_model2.alpha[i] * (tf) ** 2)
        axs[j][i].plot(tf, y.T, c="red", alpha=0.5)
        axs[j][i].set_title("$G_{%s, %s}$" % (i + 1, j + 1))
    for k in range(var_model2.C[i], max(var_model2.C)):
        axs[k][i].axis("off")
plt.savefig('filters.png', dpi=500)
plt.show()
"""

# %%
