import sherpa
from nvkm.models import MOVarNVKM
from nvkm.utils import l2p, NMSE, make_zg_grids

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
import pandas as pd
import argparse

Nbatch = 50
Nbasis = 30
noise = 0.05
Nits = 500
Nvu = 100
Ns = 5
lr = 1e-2
q_frac = 0.6
f_name = "fx"
data_dir = "data"
Nvgs = [20]
zgran = [0.75]
ampgs = [5.0]
zuran = 2.0
key = 1

keys = jrnd.split(jrnd.PRNGKey(key), 5)

client = sherpa.Client()
trial = client.get_trial()

train_df = pd.read_csv(data_dir + "/fx/fx_train.csv", index_col=0)
test_df = pd.read_csv(data_dir + "/fx/fx_test.csv", index_col=0)


def make_data(df):
    xs = []
    ys = []
    o_names = []
    y_stds = []
    y_means = []
    x = jnp.array((df["year"] - df["year"].mean()) / df["year"].std())
    for key in df.keys():
        if key != "year":
            print(key)
            o_names.append(key)
            yi = jnp.array(df[key])
            xs.append(x[~jnp.isnan(yi)])
            ysi = yi[~jnp.isnan(yi)]
            ys.append((ysi - jnp.mean(ysi)) / jnp.std(ysi))
            y_stds.append(jnp.std(ysi))
            y_means.append(jnp.mean(ysi))

    return (xs, ys), o_names, (y_means, y_stds), (df["year"].mean(), df["year"].std())


train_data, o_names, y_mean_stds, x_meanstds = make_data(train_df)

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

model.fit(
    Nits, lr, Nbatch, Ns, dont_fit=["lsgs", "ampu", "lsu", "noise"], key=keys[1],
)
print(model.noise)
print(model.ampu)
print(model.lsu)
print(model.ampgs)
print(model.lsgs)

train_preds = model.sample(train_data[0], 10)
train_nmses


tt = [jnp.array([0.0])] * O
for name in ["USD/CAD", "USD/JPY", "USD/AUD"]:
    na = ~test_df[name].isna()
    tt[o_names.index(name)] = (
        jnp.array(test_df["year"][na]) - x_meanstds[0]
    ) / x_meanstds[1]

preds = model.sample(tt, 30, key=keys[4])

nmset = 0.0
for i, key in enumerate(["USD/CAD", "USD/JPY", "USD/AUD"]):
    idx = o_names.index(key)
    pi = preds[idx] * y_mean_stds[1][idx] + y_mean_stds[0][idx]
    pred_mean = jnp.mean(pi, axis=1)
    pred_std = jnp.std(pi, axis=1)
