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

Nbatch = 50
Nbasis = 30
noise = 0.05
Nits = 500
Nvu = 80
Ns = 10
lr = 1e-3
q_frac = 0.8
f_name = "vdp"
Nvgs = [8, 8, 7]
zgran = [1.0, 1.0, 1.0]
ampgs = [0.4, 0.4, 0.4]
zuran = 20
rep = 5
mus = [2.0]
data_dir = "data"
preds_dir = "preds/duffing"

keys = jrnd.split(jrnd.PRNGKey(rep), 5)
#%%

gp1D = EQApproxGP(z=None, v=None, amp=1.0, ls=0.5, noise=0.0001, N_basis=50, D=1)


@jit
def gp_forcing(t):
    return gp1D.sample(t, 1, key=keys[0]).flatten()


@jit
def G1(x, a=1.0, b=1.0, alpha=2):
    return jnp.exp(-alpha * x ** 2) * (jnp.sin(2 * x))


@jit
def G2(x, a=1.0, b=1.0, alpha=2):
    return jnp.exp(-alpha * x ** 2) * (jnp.sin(3 * x - 2) ** 2)


@jit
def G3(x, a=1.0, b=1.0, alpha=2):
    return jnp.exp(-alpha * x ** 2) * (jnp.cos(-2 * x))


@partial(jit, static_argnums=(1, 2, 3))
def trapz_int(t, h, x, N, dim=1, decay=4):
    tau = jnp.linspace(t - decay, t + decay, N)
    ht = h(t - tau)
    xt = x(tau)
    return jnp.trapz(ht * xt, x=tau, axis=0)


N = 500
t = jnp.linspace(-20, 20, N)
Nint = 100

fyc1 = jit(lambda x: trapz_int(x, G1, gp_forcing, Nint, decay=3.0))
fyc2 = jit(lambda x: trapz_int(x, G2, gp_forcing, Nint, decay=3.0))
fyc3 = jit(lambda x: trapz_int(x, G3, gp_forcing, Nint, decay=3.0))

yc1 = vmap(fyc1)(t)
yc2 = vmap(fyc2)(t) ** 2
yc3 = vmap(fyc3)(t) ** 3
y = yc2 + yc3

u = jnp.array([gp_forcing(ti) for ti in t])
# u = (u - jnp.mean(u)) / jnp.std(u)

# pd.DataFrame({"x_train": y_train, "x_test": x_test})
# %%
plt.plot(t, yc1)
plt.plot(t, yc2)
plt.plot(t, yc3)
plt.show()
plt.plot(t, y)
plt.plot(t, u)
# %%
tg = jnp.linspace(-3, 3, 100)
plt.plot(tg, G1(tg))
plt.plot(tg, G2(tg) * G2(tg))
plt.plot(tg, G3(tg) * G3(tg) * G3(tg))
# %%
keys = jrnd.split(keys[0], 5)  #%%
O = 1
C = len(Nvgs)
zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
lsu = zu[1][0] - zu[0][0]

tgs, lsgs = make_zg_grids(zgran, Nvgs)

model = MOVarNVKM(
    [tgs] * O,
    zu,
    ([t], [y]),
    q_pars_init=None,
    q_initializer_pars=q_frac,
    q_init_key=keys[0],
    lsgs=[lsgs] * O,
    ampgs=[ampgs] * O,
    noise=[noise] * O,
    alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]] * O,
    lsu=lsu,
    ampu=1.0,
    N_basis=Nbasis,
)

# %%
model.fit(
    Nits, lr, Nbatch, Ns, dont_fit=["ampu", "lsu", "noise"], key=keys[1],
)
# %%
axs = model.plot_samples(
    jnp.linspace(-zuran, zuran, 300),
    [jnp.linspace(-zuran, zuran, 300)] * O,
    10,
    return_axs=True,
    key=keys[3],
)

plt.show()
# %%
model.plot_filters(
    jnp.linspace(-max(zgran), max(zgran), 100), 10, key=keys[3],
)

# %%
