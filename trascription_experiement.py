#%%
from nvkm.models import MOVarNVKM, EQApproxGP
from nvkm.utils import l2p, NMSE, make_zg_grids, gaussian_NLPD
from nvkm.experiments import load_duffing_data

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap
import pandas as pd
import argparse
from functools import partial
import scipy as osp

import pickle
from pathlib import Path


#%%
Nbatch = 50
Nbasis = 30
noise = 0.03
Nits = 500
Nvu = 40
Ns = 10
lr = 1e-3
q_frac = 0.8
f_name = "vdp"
Nvgs = [5, 5, 5]
zgran = [0.7, 0.7, 0.7]
ampgs = [0.7, 0.7, 0.7]
rep = 0
mus = [2.0]
data_dir = "data"
preds_dir = "preds/duffing"
zuran = 20

keys = jrnd.split(jrnd.PRNGKey(rep), 5)


gp1D = EQApproxGP(z=None, v=None, amp=1.0, ls=1.0, noise=0.0001, N_basis=50, D=1)


@jit
def gp_forcing(t):
    return gp1D.sample(t, 1, key=jrnd.PRNGKey(1)).flatten()


@jit
def G1(x):
    return jnp.exp(-(x ** 2))


@partial(jit, static_argnums=(1, 2, 3))
def trapz_int(t, h, x, N, dim=1, decay=4):
    tau = jnp.linspace(t - decay, t + decay, N)
    ht = h(t - tau)
    xt = x(tau)
    return jnp.trapz(ht * xt, x=tau, axis=0)


fyc1 = jit(lambda x: trapz_int(x, G1, gp_forcing, Nint, decay=3.0))

N = 1000
t = jnp.linspace(-20, 20, N)
Nint = 100
y = vmap(gp_forcing)(t)[:, 0] ** 2


# %%
plt.plot(t, y)
# %%
# %%
keys = jrnd.split(keys[0], 5)  #%%
O = 1
C = len(Nvgs)
zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
lsu = zu[1][0] - zu[0][0]

tgs, lsgs = make_zg_grids(zgran, Nvgs)

models = [None] * 3
for i in range(3):
    models[i] = MOVarNVKM(
        [tgs[: i + 1]] * O,
        zu,
        ([t], [y]),
        q_pars_init=None,
        q_initializer_pars=q_frac,
        q_init_key=keys[2],
        lsgs=[lsgs[: i + 1]] * O,
        ampgs=[ampgs[: i + 1]] * O,
        noise=[noise] * O,
        alpha=[[3 / (zgran[i]) ** 2 for i in range(i + 1)]] * O,
        lsu=lsu,
        ampu=1.0,
        N_basis=Nbasis,
    )


# %%
[
    model.fit(3000, 5e-3, Nbatch, Ns, dont_fit=["lsu", "noise"], key=keys[1],)
    for model in models
]


with open("squared_models.pkl", "wb") as f:
    pickle.dump(models, f)

# %%

# with open("squared_models.pkl", "rb") as f:
#     models = pickle.load(f)

axs = models[0].plot_samples(
    jnp.linspace(-zuran, zuran, 300),
    [jnp.linspace(-zuran, zuran, 300)] * O,
    10,
    return_axs=True,
    key=keys[2],
)

# %%

# %%

# %%
min_point = t[jnp.argmin(y)]
fig = plt.figure(figsize=(6, 4))
for j in range(3):
    all_s = jnp.array([])
    for i in range(500):
        all_s = jnp.hstack(
            (
                all_s,
                models[j]
                .sample(jnp.array([[min_point]]), 100, key=jrnd.PRNGKey(i))[0]
                .flatten(),
            )
        )

    plt.hist(all_s, bins=100, histtype="step", density=True, label=f"$C={j}$")
# plt.yscale("log")
plt.xlabel("Prediction at min.")
plt.legend()
plt.show()
# %%
# %%
