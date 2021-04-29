#%%
import argparse
from jax.config import config


config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import pandas as pd
import matplotlib.pyplot as plt

from wbml.data.eeg import load
from nvkm.models import MOVarNVKM
from nvkm.utils import l2p

parser = argparse.ArgumentParser(description="EEG MO experiment.")
parser.add_argument("--Nvu", default=10, type=int)
parser.add_argument("--Nvg1", default=2, type=int)
parser.add_argument("--Nvg2", default=2, type=int)
parser.add_argument("--Nits", default=1000, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--Nbatch", default=30, type=int)
parser.add_argument("--Nbasis", default=100, type=int)
parser.add_argument("--Ns", default=20, type=int)
parser.add_argument("--q_frac", default=0.5, type=float)
parser.add_argument("--noise", default=0.003, type=float)
parser.add_argument("--f_name", default="ncmogp", type=str)
parser.add_argument("--data_dir", default="data", type=str)
args = parser.parse_args()

Nbatch = args.Nbatch
Nbasis = args.Nbasis
noise = args.noise
Nits = args.Nits
Nvu = args.Nvu
Nvg1 = args.Nvg1
Nvg2 = args.Nvg2
Ns = args.Ns
lr = args.lr
q_frac = args.q_frac
f_name = args.f_name
data_dir = args.data_dir
# Nbatch = 5
# Nbasis = 30
# noise = 0.1
# Nits = 500
# Nvu = 100
# Nvg1 = 15
# Nvg2 = 6
# Ns = 5
# lr = 5e-4
# q_frac = 0.2
# f_name = "eegdev"
# data_dir = "data"
#%%

train_df = pd.read_csv(data_dir + "/eeg/eeg_train.csv")
test_df = pd.read_csv(data_dir + "/eeg/eeg_test.csv")


def make_data(df):
    xs = []
    ys = []
    o_names = []
    y_stds = []
    x = jnp.array(df["time"])
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

t1 = jnp.linspace(0.03, -0.03, Nvg1).reshape(-1, 1)
tf = jnp.linspace(0.03, -0.03, Nvg2)
tm2 = jnp.meshgrid(tf, tf)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T

# %%
model = MOVarNVKM(
    [[t1, t2]] * O,
    jnp.linspace(-0.1, 1.1, Nvu).reshape(-1, 1),
    train_data,
    q_pars_init=None,
    q_initializer_pars=q_frac,
    lsgs=[[0.008, 0.008]] * O,
    ampgs=[[4.0, 4.0]] * O,
    noise=[noise] * O,
    alpha=[l2p(0.012)] * O,
    lsu=0.02,
    ampu=10.0,
    N_basis=Nbasis,
)

#%%

model.fit(Nits, lr, Nbatch, Ns, dont_fit=["lsu", "noise"])
print(model.noise)
print(model.ampu)
print(model.lsu)
print(model.ampgs)
print(model.lsgs)
# %%
model.plot_samples(
    jnp.linspace(-0.1, 1.1, 300),
    [jnp.linspace(-0.0, 1.1, 300)] * O,
    Ns,
    save=f_name + "fit_samples.pdf",
)
model.plot_filters(jnp.linspace(0.04, -0.04, 60), 10, save=f_name + "fit_filters.pdf")
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
