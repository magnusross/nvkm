#%%
from nvkm.models import MOVarNVKM
from nvkm.utils import l2p, NMSE, make_zg_grids

from wbml.data.exchange import load
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
import argparse

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
# Nvgs = [20, 8]
# zgran = [0.75, 0.15]
# ampgs = [5.0, 5.0]
# zuran = 2.0
# key = 1

keys = jrnd.split(jrnd.PRNGKey(key), 5)
#%%
_, train_df, test_df = load(nguyen=True)


# df.to_csv("data/fx.csv")
train_df["time"] = train_df.index
test_df["time"] = test_df.index


def make_data(df):
    xs = []
    ys = []
    o_names = []
    y_stds = []
    y_means = []
    x = jnp.array(df["time"] - df["time"].mean()) / df["time"].std()
    for key in df.keys():
        if key != "time":
            o_names.append(key)
            yi = jnp.array(df[key])
            xs.append(x[~jnp.isnan(yi)])
            ysi = yi[~jnp.isnan(yi)]
            ys.append((ysi - jnp.mean(ysi)) / jnp.std(ysi))
            y_stds.append(jnp.std(ysi))
            y_means.append(jnp.mean(ysi))

    return (xs, ys), o_names, (y_means, y_stds), (df["time"].mean(), df["time"].std())


train_data, o_names, y_mean_stds, x_meanstds = make_data(train_df)

# %%
O = len(o_names)
C = len(Nvgs)

zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
lsu = zu[1][0] - zu[0][0]

tgs, lsgs = make_zg_grids(zgran, Nvgs)


# %%
model = MOVarNVKM(
    [tgs] * O,
    zu,
    train_data,
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
    300, lr, Nbatch, Ns, dont_fit=["lsgs", "ampu", "lsu", "noise"], key=keys[1],
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
    na = ~test_df[name].isna()
    idx = o_names.index(name)
    axs[idx + 1].scatter(
        (jnp.array(test_df.index[na]) - x_meanstds[0]) / x_meanstds[1],
        (test_df[name][na] - y_mean_stds[0][idx]) / y_mean_stds[1][idx],
        c="red",
        s=5.0,
        alpha=0.3,
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
# %%
tt = [jnp.array([0.0])] * O
for name in ["USD/CAD", "USD/JPY", "USD/AUD"]:
    na = ~test_df[name].isna()
    tt[o_names.index(name)] = (
        jnp.array(test_df.index[na]) - x_meanstds[0]
    ) / x_meanstds[1]

preds = model.sample(tt, 30, key=keys[4])
#%%
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
nmset = 0.0
for i, key in enumerate(["USD/CAD", "USD/JPY", "USD/AUD"]):
    idx = o_names.index(key)
    pi = preds[idx] * y_mean_stds[1][idx] + y_mean_stds[0][idx]
    pred_mean = jnp.mean(pi, axis=1)
    pred_std = jnp.std(pi, axis=1)
    y_true = jnp.array(test_df[key][~test_df[key].isna()])

    axs[i].plot(tt[idx], pred_mean, c="green", label="Pred. Mean")
    axs[i].fill_between(
        tt[idx],
        pred_mean + 2 * pred_std,
        pred_mean - 2 * pred_std,
        alpha=0.1,
        color="green",
        label="$\pm 2 \sigma$",
    )
    axs[i].plot(tt[idx], y_true, c="black", ls=":", label="Val. Data")
    axs[i].set_ylabel(key + " (V)")
    axs[i].set_xlabel(" Time (s)")
    nmse = NMSE(pred_mean, y_true)
    nmset += nmse
    print(key + f" NMSE: {nmse:.3f}")
plt.tight_layout()
axs[0].legend()
plt.tight_layout()
plt.savefig(f_name + "main.pdf")
plt.show()
print(f"Total SMSE: {nmset/3:.3f}")
# %%

