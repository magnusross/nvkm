#%%
from nvkm.models import NVKM, NewVarNVKM
from nvkm.vi import IndependentGaussians
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

keys = jrnd.split(jrnd.PRNGKey(5), 10)
# t = jnp.linspace(-20, 20, 200)

# t1 = jnp.linspace(-2.0, 2, 10).reshape(-1, 1)
# t2 = 2 * jrnd.uniform(keys[0], shape=(5, 2)) - 1.0
# t3 = 2 * jrnd.uniform(keys[0], shape=(5, 3)) - 1.0
# model = NVKM(
#     zu=jnp.linspace(-10, 10, 20).reshape(-1, 1),
#     vu=jrnd.normal(keys[1], shape=(20,)),
#     zgs=[t1, t2, t3],
#     vgs=[jnp.sin(t1).flatten(), jnp.sin(t2[:, 0] ** 2), jnp.sin(t3[:, 0] ** 2)],
#     lsgs=[1.0, 2.0, 1.0],
#     ampgs=[1.0, 1.0, 1.0],
#     C=3,
# )

# #


# # %%
# test = model.sample(t, N_s=2)

# #%%
# fig = plt.figure(figsize=(10, 5))
# plt.plot(t, test, label="Sample")
# plt.legend()
# plt.show()

# # %%
# tg = jnp.linspace(-3, 3, 100)
# for gp in model.g_gps:
#     samps = gp.sample(jnp.vstack((tg for i in range(gp.D))).T, Ns=20)
#     window = jnp.exp(-model.alpha * (tg) ** 2)

#     plt.plot(tg, (samps.T * window).T)
#     plt.show()
# #%%
# fig = plt.figure(figsize=(10, 5))
# u_samps = model.u_gp.sample(t, Ns=20)
# plt.plot(t, u_samps)
# plt.scatter(model.u_gp.z, model.u_gp.v)
t1 = jnp.linspace(-2.0, 2, 2).reshape(-1, 1)
t2 = 2 * jrnd.uniform(keys[0], shape=(2, 2)) - 1.0
t3 = 2 * jrnd.uniform(keys[0], shape=(2, 3)) - 1.0

x = jnp.linspace(-5, 5, 1)
data = (x, jnp.sin(x))
q_pars_init3 = {
    "LC_gs": [jnp.eye(2) for i in range(3)],
    "mu_gs": [jnp.sin(t1).flatten(), jnp.sin(t2[:, 0] ** 2), jnp.sin(t3[:, 0] ** 2)],
    "LC_u": jnp.eye(2),
    "mu_u": jrnd.normal(keys[1], shape=(2,)),
}
var_model3 = NewVarNVKM(
    [t1, t2, t3],
    jnp.linspace(-5, 5, 2).reshape(-1, 1),
    data,
    q_pars_init=q_pars_init3,
    lsgs=[1.0, 2.0, 1.0],
    ampgs=[1.0, 1.0, 1.0],
    noise=0.01,
)

print(var_model3.__dict__)
var_model3.sample_diag_g_gps
# %timeit model.sample(t, N_s=1)
#%%
# %timeit
# # %%
# fig = plt.figure(figsize=(10, 5))
# plt.plot(t, var_samps, label="Sample")
# plt.legend()
# plt.show()

# # %%
# q_pars_init1 = {
#     "LC_gs": [model.g_gps[0].LKvv],
#     "mu_gs": [jnp.sin(t1).flatten()],
#     "LC_u": model.u_gp.LKvv,
#     "mu_u": jrnd.normal(keys[1], shape=(20,)),
# }
# var_model1 = VariationalNVKM(
#     [t1],
#     jnp.linspace(-10, 10, 20).reshape(-1, 1),
#     None,
#     IndependentGaussians,
#     q_pars_init=q_pars_init1,
#     lsgs=[1.0],
#     ampgs_init=[1.0],
#     noise_init=0.01,
#     C=1,
# )
# # %%
# t = jnp.linspace(-20, 20, 50)
# # %timeit var_model1._sample(t, var_model1.q_of_v.q_pars, var_model1.ampgs, N_s=2).block_until_ready()
# # %%
# t = jnp.linspace(-20, 20, 300)
# samps1 = var_model1._var_sample(t, var_model1.q_of_v.q_pars, var_model1.ampgs, N_s=2)
# fig = plt.figure(figsize=(10, 5))
# plt.plot(t, samps1, label="Sample")
# plt.legend()
# plt.show()
# # %%
# from jax import grad

# grad(
#     lambda x: jnp.sum(var_model1._var_sample(t, x, var_model1.ampgs, N_s=2)), argnums=0
# )(var_model1.q_of_v.q_pars)
# # %%
