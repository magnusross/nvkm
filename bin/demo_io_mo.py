#%%
from nvkm.models import IOMOVarNVKM
from jax.config import config
import numpy as onp
from scipy.integrate import quad, dblquad

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
from functools import partial

#%%


# %%
def u(x):
    return jnp.cos(3 * x) * jnp.sin(2 * x) + jnp.cos(4.5 * x)


def G1(x, a=3.0, b=0.1, alpha=1.0):
    return jnp.exp(-alpha * x ** 2) * (-onp.sin(a * x) + b - b * x ** 2)


def G2(x, a=1.5, b=0.6, alpha=1.0):
    return jnp.exp(-alpha * x ** 2) * (-onp.cos(a * x) ** 2 + b + (x - b) ** 2)


def trapz_int(t, h, x, dim=1, decay=4, N=100):
    tau = jnp.linspace(t - decay, t + decay, N)
    ht = h(t - tau)
    xt = x(tau)
    return jnp.trapz(ht * xt, x=tau, axis=0)


noise = 0.1
Ndu = 500

xu = jnp.linspace(-15, 15, Ndu)
yu = u(xu) + noise * jrnd.normal(jrnd.PRNGKey(2), (Ndu,))

Ndy = 500
x = jnp.linspace(-15, 15, Ndy)
y1 = trapz_int(x, G1, u, decay=4.0, N=500) + noise * jrnd.normal(
    jrnd.PRNGKey(3), (Ndu,)
)
y2 = trapz_int(x, G2, u, decay=4.0, N=500) + noise * jrnd.normal(
    jrnd.PRNGKey(4), (Ndu,)
)
# y +=
# jrnd.normal(jrnd.PRNGKey(3), (Ndu,))
N_train = 300
fig = plt.figure(figsize=(30, 3))
plt.scatter(xu, yu)
plt.show()
fig = plt.figure(figsize=(30, 3))
plt.scatter(x[:N_train], y1[:N_train])
plt.scatter(x[:N_train], y2[:N_train])
plt.scatter(x[N_train:], y1[N_train:])
plt.scatter(x[N_train:], y2[N_train:])
plt.show()
plt.plot(xu, G1(xu))
plt.plot(xu, G2(xu))
plt.show()
# %%

train_data = ([x[:N_train], x[:N_train]], [y1[:N_train], y2[:N_train]])
test_data = ([x[N_train:], x[N_train:]], [y1[N_train:], y2[N_train:]])
# %%
tg = jnp.linspace(-2, 2, 20)
tu = jnp.linspace(-16, 16, 70).reshape(-1, 1)
model = IOMOVarNVKM(
    [[tg], [tg]],
    tu,
    (xu, yu),
    train_data,
    q_pars_init=None,
    q_initializer_pars=0.4,
    lsgs=[[0.4], [0.4]],
    ampgs=[[0.7], [0.7]],
    alpha=[0.5, 0.5],
    lsu=0.5,
    ampu=1.0,
    N_basis=50,
    u_noise=noise,
    noise=[noise, noise],
)


# %%
model.fit(500, 5e-3, 30, 20, dont_fit=["noise", "u_noise"])

model.save("demo_io_model.pkl")
# %%
tp = jnp.linspace(-16, 16, 150)
ul = u(tp)
yl1 = trapz_int(tp, G1, u, decay=3, N=100)
yl2 = trapz_int(tp, G2, u, decay=3, N=100)
axs = model.plot_samples(tp, [tp, tp], 10, return_axs=True)
axs[0].plot(tp, ul, c="black", ls=":", label="True $u$")
axs[1].plot(tp, yl1, c="black", ls=":", label="True $y_1$")
axs[2].plot(tp, yl2, c="black", ls=":", label="True $y_2$")
plt.show()

# %%
tf = jnp.linspace(-2.5, 2.5, 100)
axs = model.plot_filters(jnp.linspace(-2.5, 2.5, 100), 3, return_axs=True)
axs[0].plot(tf, G1(tf), label="Truth")
axs[1].plot(tf, G2(tf), label="Truth")
plt.legend()
# %%
