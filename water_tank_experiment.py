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

parser = argparse.ArgumentParser(description="Water tank experiment")
parser.add_argument("--Nvgs", default=[15, 7, 4], nargs="+", type=int)
parser.add_argument("--zgranges", default=[0.3, 0.3, 0.15], nargs="+", type=float)
parser.add_argument("--Nits", default=1000, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--noise", default=0.05, type=float)
parser.add_argument("--Nbatch", default=30, type=int)
# parser.add_argument("--Nbasis", default=100, type=int)
# parser.add_argument("--Ns", default=20, type=int)
# parser.add_argument("--q_frac", default=0.5, type=float)
# parser.add_argument("--fit_noise", default=0, type=int)
parser.add_argument("--f_name", default="ncmogp", type=str)
parser.add_argument("--data_dir", default="data", type=str)
args = parser.parse_args()

# data_dir = "data"
# Nits = 100
# Nbatch = 30
# lr = 1e-3
# f_name = "dev"
# Nvgs = [15, 7, 4]
# zran = [0.3, 0.3, 0.15]
# noise = 0.05
data_dir = args.data_dir
Nits = args.Nits
Nbatch = args.Nbatch
lr = args.lr
f_name = args.f_name
noise = args.noise
Nvgs = args.Nvgs
zran = args.zgrange


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
    (jnp.linspace(-2.0, 2.0, 150), t_offset + jnp.linspace(-2.0, 2.0, 150))
).reshape(-1, 1)
tgs = []
for i in range(C):
    tg = jnp.linspace(-zran[i], zran[i], Nvgs[i])
    tm2 = jnp.meshgrid(*[tg] * (i + 1))
    tgs.append(jnp.vstack([tm2[k].flatten() for k in range(i + 1)]).T)


# %%
modelc2 = IOMOVarNVKM(
    [tgs],
    zu,
    udata,
    ytrain,
    q_pars_init=None,
    q_initializer_pars=0.4,
    lsgs=[[0.05] * C],
    ampgs=[[7.0] * C],
    alpha=[l2p(0.1)],
    lsu=0.03,
    ampu=1.0,
    N_basis=30,
    u_noise=noise,
    noise=[noise],
)
#%%
# 5e-4
modelc2.fit(Nits, lr, Nbatch, 15, dont_fit=["lsu", "noise", "u_noise"])
modelc2.save(f_name + "tank_model.pkl")
# %%
tp_train = jnp.linspace(-2, 2, 400)
tp_test = tp_train + t_offset
axs = modelc2.plot_samples(tp_train, [tp_train], 10, return_axs=True,)
axs[0].set_xlim([-2, 2])
axs[1].set_xlim([-2, 2])
plt.savefig(f_name + "samps_train.pdf")
#%%
axs = modelc2.plot_samples(tp_test, [tp_test], 10, return_axs=True)
axs[0].set_xlim([18, 22])
axs[1].set_xlim([18, 22])
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
tf = jnp.linspace(-0.25, 0.25, 100)
modelc2.plot_filters(tf, 10, save=f_name + "filts.pdf")
# %%
