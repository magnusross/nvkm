#%%
import argparse

from jax.config import config

from nvkm.models import MultiOutputNVKM

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Compare CPU and GPU times.")
parser.add_argument("--Nvu", default=10, type=int)
parser.add_argument("--Nvg", default=2, type=int)
parser.add_argument("--zgrange", default=0.1, type=float)
parser.add_argument("--lsg", default=0.15, type=float)
parser.add_argument("--ampg", default=0.1, type=float)
parser.add_argument("--lsu", default=0.1, type=float)
parser.add_argument("--ampu", default=1.0, type=float)
parser.add_argument("--alpha", default=15.0, type=float)
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

y_raw = jnp.load("data/ncmogp/toy_y.npy")
x_raw = jnp.load("data/ncmogp/toy_x.npy")
#%%
N_train = 50
rnd_idx = jrnd.choice(jrnd.PRNGKey(12), y_raw.shape[0], shape=(N_train,))
x_train = [
    jnp.array(
        x_raw[:, i][jrnd.choice(jrnd.PRNGKey(i), y_raw.shape[0], shape=(N_train,))]
    )
    for i in range(3)
]
x_test = [
    jnp.array(
        x_raw[:, i][~jrnd.choice(jrnd.PRNGKey(i), y_raw.shape[0], shape=(N_train,))]
    )
    for i in range(3)
]
y_train = [
    jnp.array(
        y_raw[:, i][jrnd.choice(jrnd.PRNGKey(i), y_raw.shape[0], shape=(N_train,))]
    )
    for i in range(3)
]
y_test = [
    jnp.array(
        y_raw[:, i][~jrnd.choice(jrnd.PRNGKey(i), y_raw.shape[0], shape=(N_train,))]
    )
    for i in range(3)
]
data = (x_train, y_train)
test_data = (x_train, y_train)
# %%

Nvu = args.Nvu
Nvg = args.Nvg
O = 3

t1 = jnp.linspace(args.zgrange, -args.zgrange, Nvg).reshape(-1, 1)
t2p = jnp.linspace(args.zgrange, -args.zgrange, int(jnp.sqrt(Nvg)))
tm2 = jnp.meshgrid(t2p, t2p)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T
zgs = [[t1, t2], [t1, t2], [t1, t2]]

var_model2 = MultiOutputNVKM(
    zgs,
    jnp.linspace(-0.1, 1.1, Nvu).reshape(-1, 1),
    data,
    q_pars_init=None,
    q_initializer_pars=args.q_frac,
    lsgs=[[args.lsg, args.lsg], [args.lsg, args.lsg], [args.lsg, args.lsg]],
    ampgs=[[args.ampg, args.ampg], [args.ampg, args.ampg], [args.ampg, args.ampg]],
    noise=[args.noise, args.noise, args.noise],
    alpha=[args.alpha, args.alpha, args.alpha],
    lsu=args.lsu,
    ampu=args.ampu,
    N_basis=args.Nbasis,
)

# %%
Nt = 100
N_s = args.Ns
t = jnp.linspace(-0.1, 1.1, Nt)
ts = [t] * O
tf = jnp.linspace(-args.zgrange - 0.3, args.zgrange + 0.3, 100)
#%%
var_model2.plot_filters(tf, N_s, save=args.f_name + "pre_filters.png")
var_model2.plot_samples(t, ts, N_s, save=args.f_name + "pre_samples.png")
# %%
dont_fit = []
if not bool(args.fit_noise):
    dont_fit.append("noise")
var_model2.fit(args.Nits, args.lr, args.Nbatch, args.Ns, dont_fit=["noise"])


var_model2.plot_filters(tf, N_s, save=args.f_name + "fit_samples.png")
var_model2.plot_samples(t, ts, N_s, save=args.f_name + "fit_samples.png")
# %%
