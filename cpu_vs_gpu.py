import argparse

parser = argparse.ArgumentParser(description="Compare CPU and GPU times.")
parser.add_argument("--device", type=str, choices=["cpu", "gpu"])
parser.add_argument("--N", default=10, type=int)
parser.add_argument("--Ns", default=2, type=int)
args = parser.parse_args()

import jax

print("Using:", args.device)
jax.config.update("jax_platform_name", args.device)

from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians
import jax.numpy as jnp
import jax.random as jrnd
from datetime import datetime


N = args.N
N_s = args.Ns
t = jnp.linspace(-20, 20, N)
keys = jrnd.split(jrnd.PRNGKey(5), 10)

t1 = jnp.linspace(-2.0, 2, 10).reshape(-1, 1)
t2 = 2 * jrnd.uniform(keys[0], shape=(5, 2)) - 1.0
t3 = 2 * jrnd.uniform(keys[0], shape=(5, 3)) - 1.0
model = NVKM(
    zu=jnp.linspace(-10, 10, 20).reshape(-1, 1),
    vu=jrnd.normal(keys[1], shape=(20,)),
    zgs=[t1, t2, t3],
    vgs=[jnp.sin(t1).flatten(), jnp.sin(t2[:, 0] ** 2), jnp.sin(t3[:, 0] ** 2)],
    lsgs=[1.0, 2.0, 1.0],
    ampgs=[1.0, 1.0, 1.0],
    C=3,
)


# %%
q_pars_init1 = {
    "LC_gs": [model.g_gps[0].LKvv],
    "mu_gs": [jnp.sin(t1).flatten()],
    "LC_u": model.u_gp.LKvv,
    "mu_u": jrnd.normal(keys[1], shape=(20,)),
}
var_model1 = VariationalNVKM(
    [t1],
    jnp.linspace(-10, 10, 20).reshape(-1, 1),
    None,
    IndependentGaussians,
    q_pars_init=q_pars_init1,
    lsgs=[1.0],
    ampgs_init=[1.0],
    noise_init=0.01,
    C=1,
)
# %%
print("Variational C=1:")
time1 = datetime.now()
var_model1._var_sample(
    t, var_model1.q_of_v.q_pars, var_model1.ampgs, N_s
).block_until_ready()
time2 = datetime.now()
print("time:", time2 - time1)
# %%
q_pars_init3 = {
    "LC_gs": [model.g_gps[i].LKvv * 0.5 for i in range(model.C)],
    "mu_gs": [jnp.sin(t1).flatten(), jnp.sin(t2[:, 0] ** 2), jnp.sin(t3[:, 0] ** 2)],
    "LC_u": model.u_gp.LKvv,
    "mu_u": jrnd.normal(keys[1], shape=(20,)),
}
var_model3 = VariationalNVKM(
    [t1, t2, t3],
    jnp.linspace(-10, 10, 20).reshape(-1, 1),
    None,
    IndependentGaussians,
    q_pars_init=q_pars_init3,
    lsgs=[1.0, 2.0, 1.0],
    ampgs_init=[1.0, 1.0, 1.0],
    noise_init=0.01,
    C=3,
)


#%%
print("Variational C=3:")
time1 = datetime.now()
var_model3._var_sample(
    t, var_model3.q_of_v.q_pars, var_model3.ampgs, N_s
).block_until_ready()
time2 = datetime.now()
print("time:", time2 - time1)
