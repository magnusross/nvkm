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
parser.add_argument("--noise", default=0.1, type=float)
args = parser.parse_args()


keys = jrnd.split(jrnd.PRNGKey(5), 10)

noise = args.noise
Nvu = args.Nvu
Nvg = args.Nvg


tf = jnp.linspace(-3, 3, int(jnp.sqrt(Nvg)))
tm2 = jnp.meshgrid(tf, tf)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T
tf = jnp.linspace(-3, 3, 100)
t1 = jnp.linspace(-3, 3, Nvg).reshape(-1, 1)

model2 = NVKM(
    zgs=[t1, t2],
    vgs=[None, None],
    zu=jnp.linspace(-7, 7, Nvu).reshape(-1, 1),
    vu=None,
    C=2,
    lsgs=args.lsgs,
    ampgs=args.ampsgs_init,
    alpha=args.alpha,
)
model2.vgs = [
    model2.g_gps[0].sample(model2.zgs[0], 1).flatten(),
    model2.g_gps[1].sample(model2.zgs[1], 1, key=jrnd.PRNGKey(1010101)).flatten(),
]
model2.vu = model2.u_gp.sample(model2.zu, 1).flatten()
model2.g_gps = model2.set_G_gps(args.ampsgs_init, args.lsgs)
model2.u_gp = model2.set_u_gp(args.ampu, args.lsu)
# plot_c2_filter_multi(model2, tf, 15, save=args.f_name + 'non_var_c2f.png', variational=False)
# plt.show()

x = jnp.linspace(-5, 5, args.Ndata)
y = model2.sample(x, N_s=1, key=keys[6]).flatten() + noise * jrnd.normal(
    keys[9], shape=(args.Ndata,)
)
data = (x, y)

# model2.plot_samples(x, 5, save=args.f_name + 'non_var_samps.png')

q_pars_init = {
    "LC_gs": [args.q_frac * gp.LKvv for gp in model2.g_gps],
    "mu_gs": model2.vgs,
    "LC_u": args.q_frac * model2.u_gp.LKvv,
    "mu_u": model2.vu,
}


var_model2 = VariationalNVKM(
    [t1, t2],
    jnp.linspace(-7, 7, Nvu).reshape(-1, 1),
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

# plot_c2_filter_multi(var_model2, tf, 10, save=args.f_name + 'var_c2f.png', variational=True)
var_model2.plot_samples(jnp.linspace(-7, 7, 50), 5)
var_model2.plot_filters(tf, 5)
#%%
# import matplotlib.pyplot as plt

# Nim = 50
# Nax = 5
# xa = jnp.linspace(-3, 3, Nim)
# tm2 = jnp.meshgrid(xa, xa)
# tv2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T
# y = (
#     jnp.exp(-var_model2.alpha * jnp.sum((tv2) ** 2, axis=1))
#     * var_model2.g_gps[1].sample(tv2, Nax ** 2).T
# ).T
# fig, axs = plt.subplots(Nax, Nax, figsize=(10, 10))
# for i in range(Nax):
#     for j in range(Nax):
#         axs[i, j].imshow(y[:, i * j].reshape(Nim, Nim), extent=[-3, 3, -3, 3])
#         axs[i, j].scatter(var_model2.zgs[1][:, 0], var_model2.zgs[1][:, 1])
# plt.show()
# plot_c2_filter_multi(var_model2, tf, 5, variational=True)
# quit()
failed_pars = var_model2.fit(
    args.Nits, args.lr, args.Nbatch, args.Ns, dont_fit=dont_fit
)
if failed_pars:
    # f_model = VariationalNVKM(
    #     [t1, t2],
    #     jnp.linspace(-15, 15, Nvu).reshape(-1, 1),
    #     data,
    #     IndependentGaussians,
    #     q_pars_init=failed_pars["q_pars"],
    #     lsgs=args.lsgs,
    #     ampgs_init=failed_pars["ampgs"],
    #     noise_init=noise,
    #     alpha=args.alpha,
    #     lsu=args.lsu,
    #     ampu=args.ampu,
    #     C=2,
    # )

    # f_model.plot_filters(tf, 10, save=args.f_name + "failed_filters.png")
    print(failed_pars)
else:
    var_model2.plot_samples(
        jnp.linspace(-5, 5, 50), 5, save=args.f_name + "fit_samps.png"
    )
    var_model2.plot_filters(tf, 5, save=args.f_name + "fit_filter.png")
