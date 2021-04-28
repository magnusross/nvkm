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

parser = argparse.ArgumentParser(description="Compare CPU and GPU times.")
parser.add_argument("--Nvu", default=10, type=int)
parser.add_argument("--Nvg", default=2, type=int)
parser.add_argument("--zgrange", default=0.1, type=float)
parser.add_argument("--alpha", default=15.0, type=float)
parser.add_argument("--Nits", default=1000, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--Nbatch", default=30, type=int)
parser.add_argument("--Nbasis", default=100, type=int)
parser.add_argument("--Ns", default=20, type=int)
parser.add_argument("--q_frac", default=0.5, type=float)
parser.add_argument("--fit_noise", default=0, type=int)
parser.add_argument("--f_name", default="ncmogp", type=str)
parser.add_argument("--data_dir", default="data", type=str)
args = parser.parse_args()

#%%
data = pd.read_csv(args.data_dir + "/water_tanks.csv")
data = (data - data.mean()) / data.std()
# %%
noise = 0.05
udata = (jnp.array(data["Ts"]), jnp.array(data["uEst"]))
ydata = ([jnp.array(data["Ts"])], [jnp.array(data["yEst"])])

zu = jnp.linspace(-1.75, 1.75, 140).reshape(-1, 1)
tg = jnp.linspace(-0.3, 0.3, 10)
tf = jnp.linspace(-0.3, 0.3, 6)
tm2 = jnp.meshgrid(tf, tf)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T

# %%
modelc2 = IOMOVarNVKM(
    [[tg, t2]],
    zu,
    udata,
    ydata,
    q_pars_init=None,
    q_initializer_pars=0.4,
    lsgs=[[0.05, 0.06]],
    ampgs=[[7.0, 7.0]],
    alpha=[l2p(0.1)],
    lsu=0.03,
    ampu=1.0,
    N_basis=30,
    u_noise=noise,
    noise=[noise],
)
#%%
# 5e-4
modelc2.fit(args.Nits, args.lr, args.Nbatch, 15, dont_fit=["lsu", "noise", "u_noise"])
modelc2.save(args.f_name + "tank_model.pkl")
# %%
tp = jnp.linspace(-2, 2, 400)
axs = modelc2.plot_samples(
    tp, [tp], 10, return_axs=False, save=args.f_name + "samps.pdf"
)
tf = jnp.linspace(-0.25, 0.25, 100)
modelc2.plot_filters(tf, 10, save=args.f_name + "filts.pdf")
# %%
