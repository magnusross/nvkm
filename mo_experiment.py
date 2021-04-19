#%%
import argparse
from nvkm.utils import generate_C2_volterra_data
from nvkm.models import MultiOutputNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

#%%
Nvu = 10
Nvg = 25
O = 2

t1 = jnp.linspace(3, -3, Nvg).reshape(-1, 1)
tf = jnp.linspace(-3, 3, int(jnp.sqrt(Nvg)))
tm2 = jnp.meshgrid(tf, tf)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T

zgs = [[t1, t2], [t1, t2]]

var_model2 = MultiOutputNVKM(
    zgs,
    jnp.linspace(-17, 17, Nvu).reshape(-1, 1),
    None,
    q_pars_init=None,
    q_initializer_pars=0.001,
    lsgs=[[1.0, 1.0], [1.0, 1.0]],
    ampgs=[[1.0, 2.0], [1.0, 2.0]],
    noise=[0.1, 0.1],
    alpha=[1.0, 1.0],
    lsu=3.0,
    ampu=1.0,
    N_basis=50,
)

Nt = 150
t = jnp.linspace(-17, 17, Nt)
ts = [t] * O
#%%
# a = var_model2.q_of_v.sample(var_model2.q_pars, 1, jrnd.PRNGKey(1))
N_s = 10
samps = var_model2.sample(ts, N_s)
# %%
fig, axs = plt.subplots(O + 1, 1, figsize=(10, 3.5 * (1 + O)))
for i in range(O):
    axs[i].set_ylabel(f"$y_{i+1}$")
    axs[i].set_xlabel("$t$")
    axs[i].plot(ts[i], samps[i], c="green", alpha=0.5)

u_samps = var_model2.sample_u_gp(jnp.linspace(-17, 17, Nt), N_s)
axs[-1].set_ylabel(f"$u$")
axs[-1].set_xlabel("$t$")
axs[-1].scatter(var_model2.zu, var_model2.q_pars["mu_u"], c="blue", alpha=0.5)
axs[-1].plot(jnp.linspace(-17, 17, Nt), u_samps, c="blue", alpha=0.5)

plt.show()

# %%
var_model2.q_pars["mu_gs"]
# %%
