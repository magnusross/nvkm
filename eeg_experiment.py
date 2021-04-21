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

parser = argparse.ArgumentParser(description="Compare CPU and GPU times.")
parser.add_argument("--Nvu", default=10, type=int)
parser.add_argument("--Nvg", default=2, type=int)
parser.add_argument("--Nits", default=1000, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--Nbatch", default=30, type=int)
parser.add_argument("--Nbasis", default=100, type=int)
parser.add_argument("--Ns", default=20, type=int)
parser.add_argument("--q_frac", default=0.5, type=float)
parser.add_argument("--fit_noise", default=0, type=int)
parser.add_argument("--noise", default=0.003, type=float)
parser.add_argument("--f_name", default="ncmogp", type=str)

args = parser.parse_args()

#%%

_, train_df, test_df = load()


def make_data(df):
    xs = []
    ys = []
    x = jnp.array(df.index)
    for key in df.keys():
        yi = jnp.array(df[key])
        xs.append(x[~jnp.isnan(yi)])
        ysi = yi[~jnp.isnan(yi)]
        ys.append(ysi / jnp.std(ysi))
    return (xs, ys)


train_data = make_data(train_df)
# %%
Nvu = args.Nvu
Nvg = args.Nvg
O = 7

t1 = jnp.linspace(0.02, -0.02, Nvg).reshape(-1, 1)
tf = jnp.linspace(0.015, -0.015, int(jnp.sqrt(Nvg)))
tm2 = jnp.meshgrid(tf, tf)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T

# %%
model = MOVarNVKM(
    [[t1, t2]] * O,
    jnp.linspace(-0.1, 1.1, Nvu).reshape(-1, 1),
    train_data,
    q_pars_init=None,
    q_initializer_pars=args.q_frac,
    lsgs=[[0.007, 0.007]] * O,
    ampgs=[[4.0, 4.0]] * O,
    noise=[0.1] * O,
    alpha=[l2p(0.01)] * O,
    lsu=0.02,
    ampu=10.0,
    N_basis=args.Nbasis,
)
#%%
model.plot_samples(
    jnp.linspace(-0.1, 1.1, 50),
    [jnp.linspace(-0.0, 1.1, 50)] * O,
    args.Ns,
    save=args.f_name + "pre_samples.png",
)
model.plot_filters(
    jnp.linspace(0.04, -0.04, 60), 10, save=args.f_name + "pre_filters.png"
)
#%%
dont_fit = []
if not bool(args.fit_noise):
    dont_fit.append("noise")
model.fit(args.Nits, args.lr, args.Nbatch, args.Ns, dont_fit=["noise"])

print(model.noise)
print(model.ampu)
print(model.lsu)
print(model.ampgs)
print(model.lsgs)
# %%
model.plot_samples(
    jnp.linspace(-0.1, 1.1, 300),
    [jnp.linspace(-0.0, 1.1, 300)] * O,
    args.Ns,
    save=args.f_name + "fit_samples.png",
)
model.plot_filters(
    jnp.linspace(0.04, -0.04, 60), 10, save=args.f_name + "fit_filters.png"
)
