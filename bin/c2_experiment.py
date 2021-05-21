import argparse

from nvkm.utils import generate_C2_volterra_data
from nvkm.models import NVKM, VariationalNVKM
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd

parser = argparse.ArgumentParser(description="Compare CPU and GPU times.")
parser.add_argument("--Nvu", default=10, type=int)
parser.add_argument("--Nvg", default=2, type=int)
parser.add_argument("--Ndata", default=10, type=int)
parser.add_argument("--lsgs", default=[1.0, 1.0], nargs="+", type=float)
parser.add_argument("--lsu", default=1.0, type=float)
parser.add_argument("--ampu", default=1.0, type=float)
parser.add_argument("--alpha", default=1.0, type=float)
parser.add_argument("--Nits", default=10, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--Nbatch", default=1, type=int)
parser.add_argument("--Nbasis", default=100, type=int)
parser.add_argument("--ampsgs_init", default=[1.0, 1.0], nargs="+", type=float)
parser.add_argument("--Ns", default=2, type=int)
parser.add_argument("--q_frac", default=0.3, type=float)
parser.add_argument("--fit_noise", default=1, type=int)
parser.add_argument("--noise", default=0.3, type=float)
parser.add_argument("--f_name", default="test", type=str)

args = parser.parse_args()


keys = jrnd.split(jrnd.PRNGKey(5), 10)
noise = args.noise
# x_tr, y_tr, x_te, y_te = generate_C2_volterra_data(key=keys[0], N_tr=args.Ndata,
#                                                      N_te=0, noise=noise)


# data = (x_tr, y_tr)

Nvu = args.Nvu
Nvg = args.Nvg
Nvg2 = int(jnp.sqrt(Nvg)) ** 2

t1 = jnp.linspace(-3, 3, Nvg).reshape(-1, 1)

tf = jnp.linspace(-3, 3, int(jnp.sqrt(Nvg)))
tm2 = jnp.meshgrid(tf, tf)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T

model2 = NVKM(
    zgs=[t1, t2],
    vgs=[jrnd.normal(keys[3], shape=(Nvg,)), jrnd.normal(keys[2], shape=(Nvg2,))],
    zu=jnp.linspace(-17, 17, Nvu).reshape(-1, 1),
    vu=jrnd.normal(keys[1], shape=(Nvu,)),
    C=2,
    lsgs=[1.0, 1.0],
    ampgs=[1.0, 1.0],
)
x = jnp.linspace(-15, 15, args.Ndata)
y = model2.sample(x, N_s=1, key=keys[6]).flatten() + noise * jrnd.normal(
    keys[9], shape=(args.Ndata,)
)
data = (x, y)

var_model2 = VariationalNVKM(
    [t1, t2],
    jnp.linspace(-17, 17, Nvu).reshape(-1, 1),
    data,
    q_pars_init=None,
    q_initializer_pars=args.q_frac,
    lsgs=args.lsgs,
    ampgs=args.ampsgs_init,
    noise=noise,
    alpha=args.alpha,
    lsu=args.lsu,
    ampu=args.ampu,
    N_basis=args.Nbasis,
)
dont_fit = []
if not bool(args.fit_noise):
    dont_fit.append("noise")
print(var_model2.C)
var_model2.plot_samples(
    jnp.linspace(-17, 17, 150), 15, save=args.f_name + "c2_samps_pre.png"
)
var_model2.plot_filters(
    jnp.linspace(-3, 3, 100), 10, save=args.f_name + "c2_filter_pre.png"
)

var_model2.fit(args.Nits, args.lr, args.Nbatch, args.Ns, dont_fit=dont_fit)
# var_model2.save(args.f_name + "model.pkl")
var_model2.plot_samples(
    jnp.linspace(-17, 17, 150), 15, save=args.f_name + "fit_samps.png"
)
var_model2.plot_filters(
    jnp.linspace(-3, 3, 100), 10, save=args.f_name + "fit_filter.png"
)
