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
noise = 0.01
Nits = 500
Nvu = 80
Ns = 10
lr = 5e-3
q_frac = 0.8
f_name = "vdp"
Nvgs = [8, 7]
zgran = [0.8, 0.8]
ampgs = [0.6, 0.6]
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
    return jnp.exp(-alpha * x ** 2) * (-jnp.sin(6 * x))


@jit
def G2(x, a=1.0, b=1.0, alpha=2):
    return jnp.exp(-alpha * x ** 2) * (jnp.sin(5 * x) ** 2)


@jit
def G3(x, a=1.0, b=1.0, alpha=2):
    return jnp.exp(-alpha * x ** 2) * (jnp.cos(-4 * x))


@partial(jit, static_argnums=(1, 2, 3))
def trapz_int(t, h, x, N, dim=1, decay=4):
    tau = jnp.linspace(t - decay, t + decay, N)
    ht = h(t - tau)
    xt = x(tau)
    return jnp.trapz(ht * xt, x=tau, axis=0)


N = 1000
t = jnp.linspace(-20, 20, N)
Nint = 100

fyc1 = jit(lambda x: trapz_int(x, G1, gp_forcing, Nint, decay=3.0))
fyc2 = jit(lambda x: trapz_int(x, G2, gp_forcing, Nint, decay=3.0))
fyc3 = jit(lambda x: trapz_int(x, G3, gp_forcing, Nint, decay=3.0))
#%%
yc1 = vmap(fyc1)(t)
yc2 = vmap(fyc2)(t)
yc3 = vmap(fyc3)(t)
# yc3 = vmap(fyc3)(t) ** 3
y = 5 * yc1 * yc2 + 5 * yc3 ** 3
y = jnp.minimum(y, 1 * jnp.ones_like(y))
# fyc3 = jit(lambda x: trapz_int(x, G3, y, Nint, decay=3.0))
# y = vmap(fyc3)(t)

# u = (u - jnp.mean(u)) / jnp.std(u)

# pd.DataFrame({"x_train": y_train, "x_test": x_test})
# %%
fig = plt.figure(figsize=(10, 3))
plt.plot(t, yc2 * yc1)
plt.plot(t, yc3 ** 3)
plt.show()
fig = plt.figure(figsize=(20, 3))
plt.scatter(t, y)
plt.show()
# %%
tg = jnp.linspace(-3, 3, 100)
plt.plot(tg, G3(tg) ** 3)
plt.plot(tg, G2(tg) * G1(tg))
# plt.plot(tg, G3(tg) * G3(tg) * G3(tg))
#%%
x_train, y_train, x_test, y_test = load_duffing_data(3, data_dir=data_dir + "/volt")
plt.scatter(x_train, y_train)
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
    ([x_train], [y_train]),
    q_pars_init=None,
    q_initializer_pars=q_frac,
    q_init_key=keys[2],
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
    500, 3e-3, Nbatch, Ns, dont_fit=["lsu", "noise"], key=keys[1],
)
#%%
model.fit(
    int(Nits / 10),
    lr,
    Nbatch,
    Ns,
    dont_fit=["q_pars", "ampgs", "lsgs", "ampu", "lsu"],
    key=keys[2],
)
#%%
axs = model.plot_samples(
    jnp.linspace(-zuran, zuran, 300),
    [jnp.linspace(-zuran, zuran, 300)] * O,
    10,
    return_axs=True,
    key=keys[3],
)
plt.savefig(f_name + "samps.pdf")
plt.show()

#%%
t = jnp.linspace(-zuran, zuran, 200)
preds = model.predict([t], 30)
mean, var = preds[0][0], preds[1][0]
fig = plt.figure(figsize=(10, 2))
plt.scatter(x_train, y_train, c="black", s=10)
plt.plot(t, mean, c="green")
plt.scatter(x_test, y_test, c="red")
plt.fill_between(
    t, mean - 2 * jnp.sqrt(var), mean + 2 * jnp.sqrt(var), alpha=0.1, color="green",
)
plt.savefig(f_name + "main.pdf")
plt.show()
#%%
model.plot_filters(
    jnp.linspace(-max(zgran), max(zgran), 100), 10, key=keys[3],
)
plt.savefig(f_name + "filts.pdf")
# %%
m, s = model.predict([x_test], 30)
# %%
test_nmse = NMSE(m[0], y_test)
test_nlpd = gaussian_NLPD(m[0], s[0], y_test)
# %%
res = {
    "test NMSE": test_nmse,
    "test NLPD": test_nlpd,
}
print(res)
with open(f_name + "res.pkl", "wb") as f:
    pickle.dump(res, f)

f_name = "rep" + str(rep) + "predictions.csv"
odir = Path(preds_dir + "/nvkmC" + str(C))
odir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {"x_test": x_test, "y_test": y_test, "pred_mean": m[0], "pred_var": s[0]}
).to_csv(odir / f_name)
# %%
