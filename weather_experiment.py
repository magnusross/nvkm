#%%
from scipy.io import loadmat
import matplotlib.pyplot as plt
from nvkm.utils import l2p, make_zg_grids, RMSE, NMSE
from nvkm.models import MOVarNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import copy

parser = argparse.ArgumentParser(description="Weather MO experiment.")
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

Nbatch = 50
Nbasis = 30
noise = 0.05
Nits = 500
Nvu = 140
Ns = 5
lr = 1e-2
q_frac = 0.6
f_name = "eegdev"
data_dir = "data"
Nvgs = [15]
zgran = [0.5]
zuran = 2.8
ampgs = 5.0


data = loadmat(data_dir + "/weatherdata.mat")


# %%

all_x = [jnp.array(x[0].flatten()) for x in data["xT"]]
all_y = [jnp.array(y[0].flatten()) for y in data["yT"]]
#%%

train_x = copy.deepcopy(all_x)
train_y = copy.deepcopy(all_y)

train_x1 = []
train_y1 = []
test_x1 = []
test_y1 = []
for i, xi in enumerate(all_x[1]):
    if not (10.2 < xi and xi < 10.8):

        train_x1.append(xi)
        train_y1.append(all_y[1][i])
    else:
        test_x1.append(xi)
        test_y1.append(all_y[1][i])


train_x2 = []
train_y2 = []
test_x2 = []
test_y2 = []
for i, xi in enumerate(all_x[2]):
    if not 13.5 < xi < 14.2:
        train_x2.append(xi)
        train_y2.append(all_y[2][i])
    else:
        test_x2.append(xi)
        test_y2.append(all_y[2][i])

train_x[1] = jnp.array(train_x1)
train_x[2] = jnp.array(train_x2)
train_y[1] = jnp.array(train_y1)
train_y[2] = jnp.array(train_y2)
#%%
means = [jnp.mean(t) for t in train_y]
stds = [jnp.std(t) for t in train_y]
s_train_x = [jnp.array(t - 12.5) for t in train_x]
s_train_y = [(t - means[i]) / stds[i] for i, t in enumerate(train_y)]

s_testx1 = jnp.array(test_x1) - 12.5
s_testx2 = jnp.array(test_x2) - 12.5
s_testy1 = (jnp.array(test_y1) - means[1]) / stds[1]
s_testy2 = (jnp.array(test_y2) - means[2]) / stds[2]


train_data = (s_train_x, s_train_y)

# %%

# %%
O = len(train_data[0])
C = len(Nvgs)

zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
lsu = zu[1][0] - zu[0][0]

tgs, lsgs = make_zg_grids(zgran, Nvgs)

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

axs = model.plot_samples(
    jnp.linspace(-zuran, zuran, 300),
    [jnp.linspace(-zuran, zuran, 300)] * O,
    Ns,
    return_axs=True,
)
axs[2].scatter(s_testx1, s_testy1, c="red", alpha=0.3)
axs[3].scatter(s_testx2, s_testy2, c="red", alpha=0.3)
plt.show()
#%%
model.plot_filters(
    jnp.linspace(-max(zgran), max(zgran), 60), 10, save=f_name + "fit_filters.pdf"
)

# %%

preds = model.sample([jnp.array([12.5]), s_testx1, s_testx2, jnp.array([12.5])], 50)

preds1 = preds[1] * stds[1] + means[1]
mean_x1 = jnp.mean(preds1, axis=1)
std_x1 = jnp.std(preds1, axis=1)

preds2 = preds[2] * stds[2] + means[2]
mean_x2 = jnp.mean(preds2, axis=1)
std_x2 = jnp.std(preds2, axis=1)

fig, axs = plt.subplots(2, 1, figsize=(10, 5))
axs[0].plot(test_x1, test_y1, c="black", ls=":", label="Val. Data")
axs[0].plot(test_x1, mean_x1, c="green", label="Pred. Mean")
axs[0].fill_between(
    test_x1,
    mean_x1 + 2 * std_x1,
    mean_x1 - 2 * std_x1,
    alpha=0.1,
    color="green",
    label="$\pm 2 \sigma$",
)
axs[0].text(10.2, 18, f"NMSE: {NMSE(mean_x1, jnp.array(test_y1)):.2f}")

axs[1].plot(test_x2, test_y2, c="black", ls=":", label="Val. Data")
axs[1].plot(test_x2, mean_x2, c="green", label="Pred. Mean")
axs[1].fill_between(
    test_x2,
    mean_x2 + 2 * std_x2,
    mean_x2 - 2 * std_x2,
    alpha=0.1,
    color="green",
    label="$\pm 2 \sigma$",
)
axs[1].text(13.5, 22, f"NMSE: {NMSE(mean_x2, jnp.array(test_y2)):.2f}")
plt.savefig(f_name + "main.pdf")
# %%
print(f"Cambermet NMSE: {NMSE(mean_x1, jnp.array(test_y1)):.2f}")
print(f"Chimet NMSE: {NMSE(mean_x2, jnp.array(test_y2)):.2f}")


# %%
