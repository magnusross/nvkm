#%%
from nvkm.utils import l2p, make_zg_grids, RMSE
from nvkm.models import IOMOVarNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import argparse
import numpy as onp

# parser = argparse.ArgumentParser(description="Water tank IO experiment.")
# parser.add_argument("--Nvu", default=150, type=int)
# parser.add_argument("--Nvgs", default=[15, 7, 4], nargs="+", type=int)
# parser.add_argument("--zgrange", default=[0.3, 0.3, 0.25], nargs="+", type=float)
# parser.add_argument("--zurange", default=2.0, type=float)
# parser.add_argument("--Nits", default=1000, type=int)
# parser.add_argument("--lr", default=1e-3, type=float)
# parser.add_argument("--Nbatch", default=30, type=int)
# parser.add_argument("--Nbasis", default=30, type=int)
# parser.add_argument("--Ns", default=15, type=int)
# parser.add_argument("--ampgs", default=[2, 30, 30], nargs="+", type=float)
# parser.add_argument("--q_frac", default=0.5, type=float)
# parser.add_argument("--noise", default=0.05, type=float)
# parser.add_argument("--f_name", default="tank", type=str)
# parser.add_argument("--key", default=1, type=int)
# parser.add_argument("--data_dir", default="data", type=str)
# args = parser.parse_args()


# Nbatch = args.Nbatch
# Nbasis = args.Nbasis
# noise = args.noise
# Nits = args.Nits
# Nvu = args.Nvu
# Nvgs = args.Nvgs
# zgran = args.zgrange
# zuran = args.zurange
# Ns = args.Ns
# lr = args.lr
# q_frac = args.q_frac
# f_name = args.f_name
# data_dir = args.data_dir
# ampgs = args.ampgs
# key = args.key
# print(args)


data_dir = "data"
Nits = 200
Nbatch = 60
lr = 1e-2
q_frac = 0.8
f_name = "drives"
Nvu = 200
Nvgs = [15, 10]
zgran = [0.3, 0.15]
a = 5.0
ampgs = [a, a]
zuran = 2.0
zgmode = "random"
noise = 0.05
Nbasis = 30
Ns = 10
key = 1
keys = jrnd.split(jrnd.PRNGKey(key), 6)
#%%
data = loadmat(data_dir + "/drive.mat")
# %%
fig = plt.figure(figsize=(10, 2))
plt.plot(jnp.arange(len(data["u1"])), data["u1"])
plt.plot(jnp.arange(len(data["u1"])), data["z1"])

y_all = data["z1"].flatten().astype(onp.float64)
u_all = data["u1"].flatten().astype(onp.float64)


split_point = 250
t = jnp.arange(len(y_all))
yotrain, yotest = y_all[:split_point], y_all[split_point:]
uotrain, uotest = u_all[:split_point], u_all[split_point:]

y_mean, y_std = yotrain.mean(), yotrain.std()
u_mean, u_std = uotest.mean(), uotest.std()
t_mean, t_std = t.mean(), t.std()
# %%
tt = jnp.array((t - t_mean) / t_std)
utrain = (tt[:split_point], jnp.array((uotrain - u_mean) / u_std))
ytrain = ([tt[:split_point]], [jnp.array((yotrain - y_mean) / y_std)])


utest = (
    tt[split_point:],
    jnp.array((uotest - u_mean) / u_std),
)
ytest = (
    [tt[split_point:]],
    [jnp.array((yotest - y_mean) / y_std)],
)

# %%
udata = (jnp.hstack((utrain[0], utest[0])), jnp.hstack((utrain[1], utest[1])))

C = len(Nvgs)

zu = jnp.hstack(
    (jnp.linspace(-zuran, zuran, Nvu), t_offset + jnp.linspace(-zuran, zuran, Nvu))
).reshape(-1, 1)

lsu = zu[0][0] - zu[1][0]


tgs, lsgs = make_zg_grids(zgran, Nvgs)

modelc2 = IOMOVarNVKM(
    [tgs],
    zu,
    udata,
    ytrain,
    q_pars_init=None,
    q_initializer_pars=q_frac,
    q_init_key=keys[0],
    lsgs=[lsgs],
    ampgs=[ampgs],
    alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]],
    lsu=lsu,
    ampu=1.0,
    N_basis=Nbasis,
    u_noise=noise,
    noise=[noise],
)
# %%
# 5e-4
modelc2.fit(
    Nits,
    lr,
    Nbatch,
    Ns,
    dont_fit=["lsgs", "lsu", "ampu", "noise", "u_noise"],
    key=keys[2],
)
print(modelc2.noise)
print(modelc2.ampu)
print(modelc2.lsu)
print(modelc2.ampgs)
print(modelc2.lsgs)
modelc2.save(f_name + "act_model.pkl")
#%%
tp_train = jnp.linspace(-zuran, zuran, 400)
axs = modelc2.plot_samples(tp_train, [tp_train], 10, return_axs=True, key=keys[2])
axs[0].set_xlim([-zuran, zuran])
axs[1].set_xlim([-zuran, zuran])
plt.savefig(f_name + "samps_train.pdf")
# %%
p_samps = modelc2.sample(ytest[0], 20, key=keys[4])
#%%
scaled_samps = p_samps[0] * y_std + y_mean
pred_mean = jnp.mean(scaled_samps, axis=1)
pred_std = jnp.std(scaled_samps, axis=1)

rmse = RMSE(pred_mean, yotest)
print(rmse)

fig = plt.figure(figsize=(6, 4))
tp = jnp.arange(512)
plt.plot(tp, yotest, c="black", ls=":", label="Val. Data")
plt.plot(tp, pred_mean, c="green", label="Pred. Mean")
plt.fill_between(
    tp,
    pred_mean + 2 * pred_std,
    pred_mean - 2 * pred_std,
    alpha=0.1,
    color="green",
    label="$\pm 2 \sigma$",
)
plt.text(0, 3, "$e_{RMS}$ = %.2f" % rmse)
plt.xlabel("time")
plt.ylabel("output")
plt.legend()
plt.savefig(f_name + "main.pdf")
plt.show()
#%%
tf = jnp.linspace(-max(zgran), max(zgran), 100)
modelc2.plot_filters(tf, 15, save=f_name + "filts.pdf", key=keys[5])

# %%
