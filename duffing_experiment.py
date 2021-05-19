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


parser = argparse.ArgumentParser(description="Duffing experiment.")
parser.add_argument("--Nvu", default=70, type=int)
parser.add_argument("--Nvgs", default=[15], nargs="+", type=int)
parser.add_argument("--zgrange", default=[0.3], nargs="+", type=float)
parser.add_argument("--zurange", default=2.0, type=float)
parser.add_argument("--Nits", default=1000, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--Nbatch", default=30, type=int)
parser.add_argument("--Nbasis", default=30, type=int)
parser.add_argument("--Ns", default=5, type=int)
parser.add_argument("--ampgs", default=[2.0], nargs="+", type=float)
parser.add_argument("--q_frac", default=0.7, type=float)
parser.add_argument("--noise", default=0.1, type=float)
parser.add_argument("--f_name", default="vdp", type=str)
parser.add_argument("--mode", default="expr", type=str)
parser.add_argument("--rep", default=0, type=int)
parser.add_argument("--mus", default=[2.0, 1.0, 0.1, 0.0], nargs="+", type=float)
parser.add_argument("--data_dir", default="data", type=str)
parser.add_argument("--preds_dir", default="preds", type=str)
args = parser.parse_args()

Nbatch = args.Nbatch
Nbasis = args.Nbasis
noise = args.noise
Nits = args.Nits
Nvu = args.Nvu
Nvgs = args.Nvgs
zgran = args.zgrange
zuran = args.zurange
Ns = args.Ns
lr = args.lr
q_frac = args.q_frac
f_name = args.f_name
ampgs = args.ampgs
rep = args.rep
mus = args.mus
mode = args.mode
data_dir = args.data_dir
preds_dir = args.preds_dir
print(args)


# Nbatch = 50
# Nbasis = 50
# noise = 0.05
# Nits = 2000
# Nvu = 50
# Ns = 10
# lr = 5e-3
# q_frac = 0.8
# f_name = "vdp"
# Nvgs = [20, 12]
# zgran = [0.7, 0.4]
# ampgs = [2.0, 3.0]
# zuran = 1.8
# rep = 1
# mus = [2.0]
# data_dir = "data"
# preds_dir = "preds/duffing"

keys = jrnd.split(jrnd.PRNGKey(rep), 5)
#%%
x_train, y_train, x_test, y_test = load_duffing_data(2, data_dir=data_dir)
#%%
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
    q_init_key=keys[0],
    lsgs=[lsgs] * O,
    ampgs=[ampgs] * O,
    noise=[noise] * O,
    alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]] * O,
    lsu=lsu,
    ampu=1.0,
    N_basis=Nbasis,
)

#%%
model.fit(
    Nits, lr, Nbatch, Ns, dont_fit=["ampu", "lsu", "noise"], key=keys[1],
)
#%%
model.fit(
    Nits / 10,
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
t = jnp.linspace(-zuran, zuran, 300)
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
m, s = model.predict([x_test], 50)
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
pd.DataFrame({"x_test": x_test, "pred_mean": m, "pred_var": s}).to_csv(odir / f_name)
