#%%
from nvkm.utils import l2p
from nvkm.models import IOMOVarNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import pandas as pd
import argparse

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
parser.add_argument("--ampgs", default=14.0, type=float)
parser.add_argument("--q_frac", default=0.5, type=float)
parser.add_argument("--noise", default=0.05, type=float)
parser.add_argument("--f_name", default="tank", type=str)
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
print(args)
# data_dir = "data"
# Nits = 100
# Nbatch = 30
# lr = 1e-3
# q_frac = 0.8
# f_name = "dev"
# Nvgs = [15, 7, 8]
# zgran = [0.5, 0.5, 0.5]
# zuran = -2.0
# noise = 0.05
# ampgs = 12.0
# Nbasis = 30
# Ns = 15


# print(args)
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

# plt.plot(*utrain)
# plt.plot(ytrain[0][0], ytrain[1][0])
# plt.show()
# plt.plot(*utest)
# plt.plot(ytest[0][0], ytest[1][0])
# plt.show()


udata = (jnp.hstack((utrain[0], utest[0])), jnp.hstack((utrain[1], utest[1])))

C = len(Nvgs)

zu = jnp.hstack(
    (jnp.linspace(-zuran, zuran, 150), t_offset + jnp.linspace(-zuran, zuran, 150))
).reshape(-1, 1)

lsu = zu[1][0] - zu[0][0]

tgs = []
lsgs = []
for i in range(C):
    tg = jnp.linspace(-zgran[i], zgran[i], Nvgs[i])
    lsgs.append(tg[1] - tg[0])
    tm2 = jnp.meshgrid(*[tg] * (i + 1))
    tgs.append(jnp.vstack([tm2[k].flatten() for k in range(i + 1)]).T)


# %%
modelc2 = IOMOVarNVKM(
    [tgs],
    zu,
    udata,
    ytrain,
    q_pars_init=None,
    q_initializer_pars=q_frac,
    lsgs=[lsgs],
    ampgs=[[ampgs] * C],
    alpha=[3 / (max(zgran) ** 2)],
    lsu=lsu,
    ampu=1.0,
    N_basis=Nbasis,
    u_noise=noise,
    noise=[noise],
)
#%%
# 5e-4
modelc2.fit(Nits, lr, Nbatch, Ns, dont_fit=["lsu", "noise", "u_noise"])
modelc2.save(f_name + "tank_model.pkl")
# %%
tp_train = jnp.linspace(-zuran, zuran, 400)
tp_test = tp_train + t_offset
axs = modelc2.plot_samples(tp_train, [tp_train], 10, return_axs=True,)
axs[0].set_xlim([-zuran, zuran])
axs[1].set_xlim([-zuran, zuran])
plt.savefig(f_name + "samps_train.pdf")
#%%
axs = modelc2.plot_samples(tp_test, [tp_test], 10, return_axs=True)
axs[0].set_xlim([t_offset - zuran, t_offset + zuran])
axs[1].set_xlim([t_offset - zuran, t_offset + zuran])
axs[1].plot(ytest[0][0], ytest[1][0], c="black", ls=":")
plt.savefig(f_name + "samps_test.pdf")
# axs[1].xrange(18, 22)
#%%
p_samps = modelc2.sample(ytest[0], 50)
#%%
scaled_samps = p_samps[0] * y_std + y_mean
pred_mean = jnp.mean(scaled_samps, axis=1)
pred_std = jnp.std(scaled_samps, axis=1)

rmse = jnp.sqrt(
    (1 / len(data["yVal"])) * jnp.sum((pred_mean - jnp.array(data["yVal"])) ** 2)
)
print(rmse)

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
plt.xlabel("time (s)")
plt.ylabel("output (V)")
plt.legend()
plt.savefig(f_name + "main.pdf")
plt.show()
#%%
tf = jnp.linspace(-max(zgran), max(zgran), 100)
modelc2.plot_filters(tf, 10, save=f_name + "filts.pdf")
# %%
