import argparse
from nvkm.utils import generate_C2_volterra_data, plot_c2_filter_multi
from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians
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
parser.add_argument("--ampsgs_init", default=[1.0, 1.0], nargs="+", type=float)
parser.add_argument("--Ns", default=2, type=int)
parser.add_argument("--q_frac", default=0.3, type=float)
parser.add_argument("--fit_noise", default=1, type=int)
parser.add_argument("--f_name", default="test", type=str)

args = parser.parse_args()


keys = jrnd.split(jrnd.PRNGKey(5), 10)
noise = 0.01

Nvu = args.Nvu
Nvg = args.Nvg

tf = jnp.linspace(-3, 3, 100)
t1 = jnp.linspace(-3, 3, Nvg).reshape(-1, 1)
t2 = 6. * jrnd.uniform(keys[0], shape=(Nvg, 2)) - 3.

model2 = NVKM(
    zgs=[t1, t2],
    vgs = [jrnd.normal(keys[3], shape=(Nvg,)), jrnd.normal(keys[2], shape=(Nvg,))],
    zu=jnp.linspace(-15, 15, Nvu).reshape(-1, 1),
    vu=jrnd.normal(keys[1], shape=(Nvu,)),
    C=2,
    lsgs=args.lsgs,
    ampgs=args.ampsgs_init,
    )
    
plot_c2_filter_multi(model2, tf, 15, save=args.f_name + 'non_var_c2f.png', variational=False)
# plt.show()

x = jnp.linspace(-15, 15, args.Ndata)
y = model2.sample(x, N_s=1, key=keys[6]).flatten() + noise*jrnd.normal(keys[9], shape=(args.Ndata,))
data = (x, y)

model2.plot_samples(x, 5, save=args.f_name + 'non_var_samps.png')

q_pars_init = {
    "LC_gs": [args.q_frac*gp.LKvv for gp in model2.g_gps],
    "mu_gs": model2.vgs,
    "LC_u": args.q_frac*model2.u_gp.LKvv,
    "mu_u": model2.vu,
}



var_model2 = VariationalNVKM(
    [t1, t2],
    jnp.linspace(-15, 15, Nvu).reshape(-1, 1),
    data,
    IndependentGaussians,
    q_pars_init=q_pars_init,
    lsgs=args.lsgs,
    ampgs_init=args.ampsgs_init,
    noise_init=noise,
    alpha=args.alpha,
    lsu=args.lsu,
    ampu=args.ampu,
    C=2,
)
dont_fit = []
if not bool(args.fit_noise):
    dont_fit.append("noise")

plot_c2_filter_multi(var_model2, tf, 10, save=args.f_name + 'var_c2f.png', variational=True)
var_model2.plot_samples(jnp.linspace(-15, 15, 250), 15, save=args.f_name + "c2_samps_pre.png")
var_model2.plot_filters(tf, 10, save=args.f_name + "c2_filter_pre.png")


var_model2.fit(args.Nits, args.lr, args.Nbatch, args.Ns, dont_fit=dont_fit)

var_model2.plot_samples(
    jnp.linspace(-15, 15, 250), 15, save=args.f_name + "fit_samps.png"
)
var_model2.plot_filters(
    tf, 10, save=args.f_name + "fit_filter.png"
)
