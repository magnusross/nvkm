import argparse
from nvkm.utils import generate_EQ_data
from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians
from jax.config import config
import jax.numpy as jnp
import jax.random as jrnd

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--Nvu", default=10, type=int)
parser.add_argument("--Nvg", default=2, type=int)

parser.add_argument("--lsgs", default=[1.0, 1.0], nargs="+", type=float)
parser.add_argument("--lsu", default=1.0, type=float)
parser.add_argument("--Nits", default=10, type=int)
parser.add_argument("--lr", default=3e-2, type=float)
parser.add_argument("--Nbatch", default=1, type=int)
parser.add_argument("--ampsgs_init", default=[1.0, 1.0], nargs="+", type=float)
parser.add_argument("--Ns", default=2, type=int)
parser.add_argument("--fit_noise", default=1, type=int)
parser.add_argument("--f_name", default="test", type=str)

args = parser.parse_args()


keys = jrnd.split(jrnd.PRNGKey(5), 10)
x_tr, y_tr, x_te, y_te = generate_EQ_data(key=keys[0])

data = (x_tr, y_tr)

Nvu = args.Nvu
Nvg = args.Nvg

t1 = jnp.linspace(-6, 6, Nvg).reshape(-1, 1)
var_model1 = VariationalNVKM(
    [t1],
    jnp.linspace(-44, 44, Nvu).reshape(-1, 1),
    data,
    IndependentGaussians,
    q_pars_init=None,
    q_initializer_pars=0.3,
    lsgs=args.lsgs,
    ampgs_init=args.ampsgs_init,
    noise_init=0.3,
    alpha=0.45,
    C=1,
)

dont_fit = []
if not bool(args.fit_noise):
    dont_fit.append("noise")


var_model1.fit(args.Nits, args.lr, args.Nbatch, args.Ns, dont_fit=dont_fit)
# var_model1.save(args.f_name + "model.pkl")
var_model1.plot_samples(
    jnp.linspace(-44, 44, 250), 10, save=args.f_name + "fit_samps.png"
)
var_model1.plot_filters(
    jnp.linspace(-6, 6, 100), 10, save=args.f_name + "fit_filter.png"
)
