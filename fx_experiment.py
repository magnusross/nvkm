#%%
from nvkm.models import MOVarNVKM
from nvkm.utils import l2p, NMSE, make_zg_grids, gaussian_NLPD
from nvkm.experiments import ExchangeDataSet
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser(description="FX MO experiment.")
parser.add_argument("--Nvu", default=60, type=int)
parser.add_argument("--Nvgs", default=[20, 8], nargs="+", type=int)
parser.add_argument("--zgrange", default=[0.75, 0.15], nargs="+", type=float)
parser.add_argument("--zurange", default=2.0, type=float)
parser.add_argument("--Nits", default=1000, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--Nbatch", default=30, type=int)
parser.add_argument("--Nbasis", default=30, type=int)
parser.add_argument("--Ns", default=5, type=int)
parser.add_argument("--ampgs", default=[5, 5], nargs="+", type=float)
parser.add_argument("--q_frac", default=0.6, type=float)
parser.add_argument("--noise", default=0.05, type=float)
parser.add_argument("--f_name", default="fx", type=str)
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

# Nbatch = 50
# Nbasis = 30
# noise = 0.05
# Nits = 500
# Nvu = 100
# Ns = 5
# lr = 1e-2
# q_frac = 0.6
# f_name = "fx"
# data_dir = "data"
# Nvgs = [20]
# zgran = [0.75]
# ampgs = [5.0]
# zuran = 2.0
# key = 1

keys = jrnd.split(jrnd.PRNGKey(key), 5)

data = ExchangeDataSet(data_dir)
# %%
O = data.O
C = len(Nvgs)

zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
lsu = zu[1][0] - zu[0][0]

tgs, lsgs = make_zg_grids(zgran, Nvgs)


# %%
model = MOVarNVKM(
    [tgs] * O,
    zu,
    (data.strain_x, data.strain_y),
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
    Nits, lr, Nbatch, Ns, dont_fit=["lsgs", "ampu", "lsu", "noise"], key=keys[1],
)
model.fit(
    int(Nits / 10),
    lr,
    Nbatch,
    Ns,
    dont_fit=["q_pars", "ampgs", "lsgs", "ampu", "lsu"],
    key=keys[5],
)
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
    key=keys[2],
)
for name in ["USD/CAD", "USD/JPY", "USD/AUD"]:
    idx = data.output_names.index(name)
    axs[idx + 1].scatter(
        data.stest_x[idx], data.stest_y[idx], c="red", s=5.0, alpha=0.3
    )
plt.savefig(f_name + "fit_samples.pdf")
plt.show()

# %%
model.plot_filters(
    jnp.linspace(-max(zgran), max(zgran), 60),
    10,
    save=f_name + "fit_filters.pdf",
    key=keys[3],
)
#%%

spreds = model.predict(data.stest_x, 5, key=keys[4])
_, pred_mean = data.upscale(data.stest_x, spreds[0])
pred_var = data.upscale_variance(spreds[1])

train_spreds = model.predict(data.strain_x, 5, key=keys[5])
_, train_pred_mean = data.upscale(data.strain_x, train_spreds[0])
train_pred_var = data.upscale_variance(train_spreds[1])

train_nmse = sum([NMSE(train_pred_mean[i], data.train_y[i]) for i in range(O)])
train_nlpd = sum(
    [
        gaussian_NLPD(train_pred_mean[i], train_pred_var[i], data.train_y[i])
        for i in range(O)
    ]
)

test_nmse = (
    sum(
        [
            NMSE(
                pred_mean[data.output_names.index(n)],
                data.test_y[data.output_names.index(n)],
            )
            for n in ["USD/CAD", "USD/JPY", "USD/AUD"]
        ]
    )
    / 3
)
test_nlpd = (
    sum(
        [
            gaussian_NLPD(
                pred_mean[data.output_names.index(n)],
                pred_var[data.output_names.index(n)],
                data.test_y[data.output_names.index(n)],
            )
            for n in ["USD/CAD", "USD/JPY", "USD/AUD"]
        ]
    )
    / 3
)
res = {
    "test NMSE": test_nmse,
    "train NMSE": train_nmse,
    "test NLPD": test_nlpd,
    "train NLPD": train_nlpd,
}
print(res)

with open(f_name + "res.pkl", "wb") as f:
    pickle.dump(res, f)

#%%
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
nmset = 0.0
for i, key in enumerate(["USD/CAD", "USD/JPY", "USD/AUD"]):
    idx = data.output_names.index(key)
    pmi = pred_mean[idx]
    psi = pred_var[idx]

    axs[i].plot(data.test_x[idx], pmi, c="green", label="Pred. Mean")
    axs[i].fill_between(
        data.test_x[idx],
        pmi - 2 * jnp.sqrt(psi),
        pmi + 2 * jnp.sqrt(psi),
        alpha=0.1,
        color="green",
        label="$\pm 2 \sigma$",
    )
    axs[i].plot(
        data.test_x[idx], data.test_y[idx], c="black", ls=":", label="Val. Data"
    )
    axs[i].set_ylabel(key)
    axs[i].set_xlabel(" Time (years)")

axs[0].legend()
plt.tight_layout()
plt.savefig(f_name + "main.pdf")
plt.show()
# %%

