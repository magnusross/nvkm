#%%
from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

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
    ampgs=[1.0, 0.0, 0.0],
    C=3,
)

# %%
t = jnp.linspace(-20, 20, 300)
test = model.sample(t, N_s=2)

#%%
fig = plt.figure(figsize=(10, 5))
plt.plot(t, test, label="Sample")
plt.legend()
plt.show()

# %%
tg = jnp.linspace(-3, 3, 100)
for gp in model.g_gps:
    samps = gp.sample(jnp.vstack((tg for i in range(gp.D))).T, Ns=20)
    window = jnp.exp(-model.alpha * (tg) ** 2)

    plt.plot(tg, (samps.T * window).T)
    plt.show()
#%%
fig = plt.figure(figsize=(10, 5))
u_samps = model.u_gp.sample(t, Ns=20)
plt.plot(t, u_samps)
plt.scatter(model.u_gp.z, model.u_gp.v)
# %%

q_pars_init = {
    "LC_gs": [model.g_gps[i].LKvv * 0.5 for i in range(model.C)],
    "mu_gs": [jnp.sin(t1).flatten(), jnp.sin(t2[:, 0] ** 2), jnp.sin(t3[:, 0] ** 2)],
    "LC_u": model.u_gp.LKvv,
    "mu_u": jrnd.normal(keys[1], shape=(20,)),
}
var_model = VariationalNVKM(
    [t1, t2, t3],
    jnp.linspace(-10, 10, 20).reshape(-1, 1),
    None,
    IndependentGaussians,
    q_pars_init=q_pars_init,
    lsgs=[1.0, 2.0, 1.0],
    ampgs_init=[1.0, 0.0, 0.0],
    noise_init=0.01,
    C=3,
)

# %%
var_model._var_sample(t, var_model.q_of_v.q_pars, var_model.ampgs)
# %%
