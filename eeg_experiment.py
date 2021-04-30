#%%
import argparse
from jax.config import config


config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import pandas as pd
import matplotlib.pyplot as plt

from nvkm.models import MOVarNVKM
from nvkm.utils import l2p

parser = argparse.ArgumentParser(description="EEG MO experiment.")
parser.add_argument("--Nvu", default=60, type=int)
parser.add_argument("--Nvgs", default=[15], nargs="+", type=int)
parser.add_argument("--zgrange", default=[0.25], nargs="+", type=float)
parser.add_argument("--zurange", default=1.8, type=float)
parser.add_argument("--Nits", default=1000, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--Nbatch", default=30, type=int)
parser.add_argument("--Nbasis", default=30, type=int)
parser.add_argument("--Ns", default=20, type=int)
parser.add_argument("--ampgs", default=1.0, type=float)
parser.add_argument("--q_frac", default=0.5, type=float)
parser.add_argument("--noise", default=0.1, type=float)
parser.add_argument("--f_name", default="eeg", type=str)
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
# Nbatch = 5
# Nbasis = 30
# noise = 0.05
# Nits = 500
# Nvu = 60
# Ns = 5
# lr = 5e-4
# q_frac = 0.8
# f_name = "eegdev"
# data_dir = "data"
# Nvgs = [15]
# zgran = [0.2]
# zuran = 1.8
#%%

train_df = pd.read_csv(data_dir + "/eeg/eeg_train.csv")
test_df = pd.read_csv(data_dir + "/eeg/eeg_test.csv")


def make_data(df):
    xs = []
    ys = []
    o_names = []
    y_stds = []
    x = jnp.array(df["time"] - df["time"].mean()) / df["time"].std()
    for key in df.keys():
        if key != "time":
            o_names.append(key)
            yi = jnp.array(df[key])
            xs.append(x[~jnp.isnan(yi)])
            ysi = yi[~jnp.isnan(yi)]
            ys.append(ysi / jnp.std(ysi))
            y_stds.append(jnp.std(ysi))
    return (xs, ys), o_names, y_stds


train_data, o_names, y_stds = make_data(train_df)
# %%

O = 7
C = len(Nvgs)

zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
lsu = zu[1][0] - zu[0][0]
tgs = []
lsgs = []
for i in range(C):
    tg = jnp.linspace(-zgran[i], zgran[i], Nvgs[i])
    lsgs.append(tg[1] - tg[0])
    tm2 = jnp.meshgrid(*[tg] * (i + 1))
    tgs.append(jnp.vstack([tm2[k].flatten() for k in range(i + 1)]).T)


# %%
model = MOVarNVKM(
    [tgs] * O,
    zu,
    train_data,
    q_pars_init=None,
    q_initializer_pars=q_frac,
    lsgs=[lsgs] * O,
    ampgs=[[ampgs] * C] * O,
    noise=[noise] * O,
    alpha=[3 / (max(zgran) ** 2)] * O,
    lsu=lsu,
    ampu=1.0,
    N_basis=Nbasis,
)

#%%

model.fit(Nits, lr, Nbatch, Ns, dont_fit=["lsgs", "lsu", "noise"])
print(model.noise)
print(model.ampu)
print(model.lsu)
print(model.ampgs)
print(model.lsgs)
# %%
model.plot_samples(
    jnp.linspace(-zuran, zuran, 300),
    [jnp.linspace(-zuran, zuran, 300)] * O,
    Ns,
    save=f_name + "fit_samples.pdf",
)
model.plot_filters(
    jnp.linspace(-max(zgran), max(zgran), 60), 10, save=f_name + "fit_filters.pdf"
)
#%%
tt = jnp.array(test_df.index)
preds = model.sample([tt] * O, 50)
#%%
fig, axs = plt.subplots(3, 1, figsize=(13, 7))
for i, key in enumerate(["FZ", "F1", "F2"]):
    idx = o_names.index(key)
    pi = preds[idx] * y_stds[idx]
    pred_mean = jnp.mean(pi, axis=1)
    pred_std = jnp.std(pi, axis=1)
    axs[i].plot(tt, pred_mean, c="green", label="Pred. Mean")
    axs[i].fill_between(
        tt,
        pred_mean + 2 * pred_std,
        pred_mean - 2 * pred_std,
        alpha=0.1,
        color="green",
        label="$\pm 2 \sigma$",
    )
    axs[i].plot(tt, test_df[key], c="black", ls=":", label="Val. Data")
    axs[i].set_ylabel(key + " (V)")
    axs[i].set_xlabel(" Time (s)")
axs[0].legend()
plt.tight_layout()
plt.savefig(f_name + "main.pdf")
plt.show()
# %%
