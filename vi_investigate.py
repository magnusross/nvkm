#%%
from nvkm import vi
import jax.numpy as jnp
import jax.random as jrnd
from nvkm.models import NVKM, VariationalNVKM
import matplotlib.pyplot as plt

#%%
keys = jrnd.split(jrnd.PRNGKey(5), 10)
noise = 0.01

Nvu = 10
Nvg = 50
Ndata = 100


tf = jnp.linspace(-3, 3, 100)
t1 = jnp.linspace(-3, 3, Nvg).reshape(-1, 1)
t2 = 6.0 * jrnd.uniform(keys[0], shape=(Nvg, 2)) - 3.0

model = NVKM(
    zgs=[t1, t2],
    vgs=[jrnd.normal(keys[3], shape=(Nvg,)), jrnd.normal(keys[2], shape=(Nvg,))],
    zu=jnp.linspace(-15, 15, Nvu).reshape(-1, 1),
    vu=jrnd.normal(keys[1], shape=(Nvu,)),
    C=2,
    lsgs=[1.0, 1.0],
    ampgs=[1.0, 1.0],
    lsu=5.0,
)

x = jnp.linspace(-15, 15, Ndata)
y = model.sample(x, N_s=1, key=keys[6]).flatten() + noise * jrnd.normal(
    keys[9], shape=(Ndata,)
)
data = (x, y)

# model2.plot_samples(x, 5, save=args.f_name + 'non_var_samps.png')
q_frac = 0.01
q_pars_init = {
    "LC_gs": [q_frac * gp.LKvv for gp in model.g_gps],
    "mu_gs": model.vgs,
    "LC_u": q_frac * model.u_gp.LKvv,
    "mu_u": model.vu,
}

#%%
var_model = VariationalNVKM(
    [t1, t2],
    jnp.linspace(-15, 15, Nvu).reshape(-1, 1),
    data,
    vi.IndependentGaussians,
    q_pars_init=q_pars_init,
    lsgs=[1.0, 1.0],
    ampgs_init=[1.0, 1.0],
    noise_init=1.0,
    alpha=1.0,
    ampu=1.0,
    lsu=5.0,
    C=2,
)

#%%
s = jrnd.normal(jrnd.PRNGKey(1), shape=(100,))
dist = vi.IndependentGaussians()
# dist.single_KL(jnp.eye(100), s, 0.9 * jnp.eye(100))
dist._KL(var_model.p_pars, var_model.q_pars)
dist.single_KL(
    var_model.q_pars["LC_gs"][0],
    model.vgs[0],  # model.sample(t1, 1, key=jrnd.PRNGKey(100101)).flatten(),
    var_model.p_pars["LK_gs"][0],
)
# %%
var_model.compute_bound(10)

# %%

samps = model.sample(x, 10)
vi.gaussain_likelihood(y, samps, noise)
plt.plot(x, y)
plt.plot(x, samps)
# %%

# %%
