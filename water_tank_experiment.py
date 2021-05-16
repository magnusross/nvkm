#%%
from nvkm.utils import l2p, make_zg_grids, RMSE, gaussian_NLPD
from nvkm.models import IOMOVarNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser(description="Water tank IO experiment.")
parser.add_argument("--Nvu", default=150, type=int)
parser.add_argument("--Nvgs", default=[15, 7, 4], nargs="+", type=int)
parser.add_argument("--zgrange", default=[0.3, 0.3, 0.25], nargs="+", type=float)
parser.add_argument("--zurange", default=2.0, type=float)
parser.add_argument("--Nits", default=1000, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--Nbatch", default=30, type=int)
parser.add_argument("--Nbasis", default=30, type=int)
parser.add_argument("--Ns", default=15, type=int)
parser.add_argument("--ampgs", default=[2, 30, 30], nargs="+", type=float)
parser.add_argument("--q_frac", default=0.5, type=float)
parser.add_argument("--noise", default=0.05, type=float)
parser.add_argument("--f_name", default="tank", type=str)
parser.add_argument("--key", default=1, type=int)
parser.add_argument("--data_dir", default="data", type=str)
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
data_dir = args.data_dir
ampgs = args.ampgs
key = args.key
print(args)
# data_dir = "data"
# Nits = 200
# Nbatch = 30
# lr = 1e-2
# q_frac = 0.55
# f_name = "dev"
# Nvgs = [15, 8]
# zgran = [0.3, 0.2]
# a = 6.0
# ampgs = [a, a, a]
# zuran = 2.0
# zgmode = "random"
# noise = 0.05
# Nbasis = 30
# Ns = 5
# key = 1


# print(args)
keys = jrnd.split(jrnd.PRNGKey(key), 7)

#%%
data = pd.read_csv(data_dir + "/water_tanks.csv")
y_mean, y_std = data["yEst"].mean(), data["yEst"].std()
u_mean, u_std = data["uEst"].mean(), data["uEst"].std()
t_mean, t_std = data["Ts"].mean(), data["Ts"].std()
# %%

tt = jnp.array((data["Ts"] - t_mean) / t_std)
utrain = (tt, jnp.array((data["uEst"] - u_mean) / u_std))
ytrain = ([tt], [jnp.array((data["yEst"] - y_mean) / y_std)])

t_offset = 20.0
utest = (
    tt + t_offset * jnp.ones(len(data)),
    jnp.array((data["uVal"] - u_mean) / u_std),
)
ytest = (
    [tt + t_offset * jnp.ones(len(data))],
    [jnp.array((data["yVal"] - y_mean) / y_std)],
)

udata = (jnp.hstack((utrain[0], utest[0])), jnp.hstack((utrain[1], utest[1])))

C = len(Nvgs)

zu = jnp.hstack(
    (jnp.linspace(-zuran, zuran, 150), t_offset + jnp.linspace(-zuran, zuran, 150))
).reshape(-1, 1)

lsu = zu[0][0] - zu[1][0]


tgs, lsgs = make_zg_grids(zgran, Nvgs)
# %%

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
#%%
# 5e-4
modelc2.fit(
    Nits, lr, Nbatch, Ns, dont_fit=["lsu", "noise", "u_noise"], key=keys[1],
)
print(modelc2.noise)
print(modelc2.ampu)
print(modelc2.lsu)
print(modelc2.ampgs)
print(modelc2.lsgs)
# %%
tp_train = jnp.linspace(-zuran, zuran, 400)
tp_test = tp_train + t_offset
axs = modelc2.plot_samples(tp_train, [tp_train], 10, return_axs=True, key=keys[2])
axs[0].set_xlim([-zuran, zuran])
axs[1].set_xlim([-zuran, zuran])
plt.savefig(f_name + "samps_train.pdf")
#%%
axs = modelc2.plot_samples(tp_test, [tp_test], 10, return_axs=True, key=keys[3])
axs[0].set_xlim([t_offset - zuran, t_offset + zuran])
axs[1].set_xlim([t_offset - zuran, t_offset + zuran])
axs[1].plot(ytest[0][0], ytest[1][0], c="black", ls=":")
plt.savefig(f_name + "samps_test.pdf")
# axs[1].xrange(18, 22)
#%%

p_samps = modelc2.sample(ytest[0], 50, key=keys[4])
p_samps_tr = modelc2.sample(ytrain[0], 50, key=keys[5])
#%%
scaled_samps = p_samps[0] * y_std + y_mean
pred_mean = jnp.mean(scaled_samps, axis=1)
pred_std = jnp.std(scaled_samps, axis=1)
pred_var = pred_std ** 2 + modelc2.noise[0] ** 2

scaled_samps_tr = p_samps_tr[0] * y_std + y_mean
pred_mean_tr = jnp.mean(scaled_samps_tr, axis=1)
pred_std_tr = jnp.std(scaled_samps_tr, axis=1)
pred_var_tr = pred_std_tr ** 2 + modelc2.noise[0] ** 2

rmse = RMSE(pred_mean, jnp.array(data["yVal"]))
nlpd = gaussian_NLPD(pred_mean, pred_var, jnp.array(data["yVal"]))

rmse_tr = RMSE(pred_mean_tr, jnp.array(data["yEst"]))
nlpd_tr = gaussian_NLPD(pred_mean_tr, pred_var_tr, jnp.array(data["yEst"]))
#%%
print("Train RMSE: %.3f" % rmse_tr)
print("Train NLPD: %.3f" % nlpd_tr)

print("RMSE: %.3f" % rmse)
print("NLPD: %.3f" % nlpd)

res = {
    "test NMSE": rmse,
    "train NMSE": rmse_tr,
    "test NLPD": nlpd,
    "train NLPD": nlpd_tr,
}

print(res)
with open(f_name + "res.pkl", "wb") as f:
    pickle.dump(res, f)

#%%
fig = plt.figure(figsize=(12, 4))
plt.plot(data["Ts"], data["yVal"], c="black", ls=":", label="Val. Data")
plt.plot(data["Ts"], pred_mean, c="green", label="Pred. Mean")
plt.fill_between(
    data["Ts"],
    pred_mean + 2 * pred_std,
    pred_mean - 2 * pred_std,
    alpha=0.1,
    color="green",
    label="$\pm 2 \sigma$",
)
plt.text(0, 12, "$e_{RMS}$ = %.2f" % rmse)
plt.text(0, 13, "$e_{NLPD}$ = %.2f" % nlpd)
plt.xlabel("time (s)")
plt.ylabel("output (V)")
plt.legend()
plt.savefig(f_name + "main.pdf")
plt.show()
#%%
tf = jnp.linspace(-max(zgran), max(zgran), 100)
modelc2.plot_filters(tf, 15, save=f_name + "filts.pdf", key=keys[6])
# %%
