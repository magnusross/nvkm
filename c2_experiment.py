import argparse
from nvkm.utils import generate_C2_volterra_data
from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians
import jax.numpy as jnp
import jax.random as jrnd

parser = argparse.ArgumentParser(description="Compare CPU and GPU times.")
parser.add_argument("--Nvu", default=10, type=int)
parser.add_argument("--Nvg", default=2, type=int)
parser.add_argument("--Ndata", default=10, type=int)
parser.add_argument("--lsgs", default=[1.0, 1.0], type=list)
parser.add_argument("--lsu", default=1.0, type=float)
parser.add_argument("--Nits", default=10, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--Nbatch", default=1, type=int)
parser.add_argument("--Ns", default=2, type=int)


args = parser.parse_args()


keys = jrnd.split(jrnd.PRNGKey(5), 10)
x_tr, y_tr, x_te, y_te = generate_C2_volterra_data(key=keys[0], N_tr=args.Ndata, N_te=0)

data = (x_tr, y_tr)

Nvu = args.Nvu
Nvg = args.Nvg


q_pars_init = {
    "LC_gs": [0.3 * jnp.eye(Nvg), 0.3 * jnp.eye(Nvg)],
    "mu_gs": [jrnd.normal(keys[1], shape=(Nvg,)), jrnd.normal(keys[2], shape=(Nvg,))],
    "LC_u": 0.3 * jnp.eye(Nvu),
    "mu_u": jrnd.normal(keys[3], shape=(Nvu,)),
}

t1 = jnp.linspace(-3, 3, Nvg).reshape(-1, 1)
t2 = 6 * jrnd.uniform(keys[4], shape=(Nvg, 2)) - 3

var_model2 = VariationalNVKM(
    [t1, t2],
    jnp.linspace(-30, 30, Nvu).reshape(-1, 1),
    data,
    IndependentGaussians,
    q_pars_init=q_pars_init,
    lsgs=args.lsgs,
    ampgs_init=[1.0, 1.0],
    noise_init=0.6,
    lsu=args.lsu,
    C=2,
)

var_model2.fit(args.Nits, args.lr, args.Nbatch, args.Ns)
var_model2.plot_samples(jnp.linspace(-30, 30, 500), args.Ns, save=True)
var_model2.plot_filters(jnp.linspace(-3, 3, 100), 10, save=True)
