#%%
import argparse
from nvkm.utils import generate_C2_volterra_data, plot_c2_filter_multi
from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians
from jax.config import config
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd

#%%

keys = jrnd.split(jrnd.PRNGKey(5), 10)
noise = 0.01

Nvu = 100
Nvg = 27
Ndata = 30
alpha = 0.45
Nplot = 100

f_name = "plots/tl"
lsgs = [1.0, 1.0]

ampsgs = [1.0, 1.0]
ampu = 1.0
lsu = 0.1
tip = jnp.linspace(-3, 3, Nvg)

tip1 = tip.reshape(-1, 1)
tf = jnp.linspace(-3, 3, int(jnp.sqrt(Nvg)))
tm2 = jnp.meshgrid(tf, tf)
tip2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T

model2 = NVKM(
    zgs=[tip1, tip2],
    vgs=[None, None],
    zu=jnp.linspace(-6, 3, Nvu).reshape(-1, 1),
    vu=None,
    C=2,
    lsgs=lsgs,
    lsu=lsu,
    ampgs=ampsgs,
    ampu=ampu,
    alpha=alpha,
    N_basis=50,
)
model2.vgs = [
    model2.g_gps[0].sample(model2.zgs[0], 1).flatten(),
    model2.g_gps[1].sample(model2.zgs[1], 1, key=jrnd.PRNGKey(1010101)).flatten(),
]
model2.vu = model2.u_gp.sample(model2.zu, 1).flatten()
model2.g_gps = model2.set_G_gps(ampsgs, lsgs)
model2.u_gp = model2.set_u_gp(ampu, lsu)

xf = jnp.linspace(-5, 5, 200)

plot_c2_filter_multi(model2, xf, 15)
# plt.show()


x = jnp.linspace(-6, 3, Nplot)

model2.plot_samples(x, 5, key=jrnd.PRNGKey(101001010))
#%%


#%%
# Nim = 50
# Nax = 5
# xa = jnp.linspace(-1, 1, Nim)
# tm2 = jnp.meshgrid(xa, xa)
# tv2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T
# y = (
# exp(-model2.alpha * jnp.sum((tv2) ** 2, axis=1))
#     * model2.g_gps[1].sample(tv2, Nax ** 2).T
# ).T
# fig, axs = plt.subplots(Nax, Nax, figsize=(10, 10))
# for i in range(Nax):
#     for j in range(Nax):
#         axs[i, j].imshow(y[:, i * j].reshape(Nim, Nim))
# plt.show()

# plt.plot(xf, model2.g_gps[1].sample(jnp.vstack((xf, xf)).T, 10))
# plt.scatter(tip, model2.vgs[1])
#%%
s2 = model2.sample(x, 5)
model2.C = 1
s1 = model2.sample(x, 5)
model2.C = 2
# %%
fig = plt.figure(figsize=(10, 5))
plt.plot(x, s2, c="red")
plt.plot(x, s1, c="green")
plt.plot(x, s2 - s1, c="blue")
plt.legend()
plt.show()

# %%

# %%
Nvu = 20
Nvg = 27
Ndata = 30
alpha = 3.0
Nplot = 100

lsgs = [1.0, 1.0, 0.1]

ampsgs = [1.0, 1.0, 10.0]
ampu = 1.0
lsu = 0.5
tip = jnp.linspace(-1, 1, Nvg)
tip1 = tip.reshape(-1, 1)
tip2 = 2 * jrnd.uniform(jrnd.PRNGKey(101), shape=(Nvg, 2)) - 1
tip3 = 2 * jrnd.uniform(jrnd.PRNGKey(103), shape=(Nvg, 3)) - 1

model3d = NVKM(
    zgs=[tip1, tip2, tip3],
    vgs=[None, None, None],
    zu=jnp.linspace(-6, 3, Nvu).reshape(-1, 1),
    vu=None,
    C=3,
    lsgs=lsgs,
    lsu=lsu,
    ampgs=ampsgs,
    ampu=ampu,
    alpha=alpha,
    N_basis=100,
)
model3d.vgs = [
    model3d.g_gps[0].sample(model3d.zgs[0], 1).flatten(),
    model3d.g_gps[1].sample(model3d.zgs[1], 1, key=jrnd.PRNGKey(1010101)).flatten(),
    model3d.g_gps[2].sample(model3d.zgs[2], 1, key=jrnd.PRNGKey(12)).flatten(),
]
model3d.vu = model3d.u_gp.sample(model3d.zu, 1).flatten()
model3d.g_gps = model3d.set_G_gps(ampsgs, lsgs)
model3d.u_gp = model3d.set_u_gp(ampu, lsu)

xf = jnp.linspace(-1, 1, 200)


#%%

# plot_c2_filter_multi(model3d, xf, 15)
# plt.show()
#%%

x = jnp.linspace(-6, 3, Nplot)
model3d.plot_samples(x, 3, key=jrnd.PRNGKey(101001010))

# %%

# %%

