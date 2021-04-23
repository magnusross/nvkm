#%%
from nvkm.models import IOMOVarNVKM
from jax.config import config
import numpy as onp
from scipy.integrate import quad, dblquad

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

#%%


# %%
def u(x):
    return jnp.cos(3 * x) + jnp.sin(5 * x)


def G1(x, alpha=1.0):
    return jnp.exp(-alpha * x ** 2) * (-onp.sin(3 * x) + 0.3 - 0.3 * x ** 2)


def G2(x, alpha=1.0):
    return jnp.exp(-alpha * x ** 2) * (-onp.cos(3 * x) ** 2 + 0.3 - 0.3 * x ** 5)


def trapz_int(t, h, x, dim=1, decay=4, N=100):
    tau = jnp.linspace(t - decay, t + decay, N)
    ht = h(t - tau)
    xt = x(tau)
    return jnp.trapz(ht * xt, x=tau, axis=0)


noise = 0.1
Ndu = 200

xu = jnp.linspace(-5, 5, Ndu)
yu = u(xu) + noise * jrnd.normal(jrnd.PRNGKey(2), (Ndu,))

Ndy = 200
x = jnp.linspace(-5, 5, Ndy)
y1 = trapz_int(x, G1, u, decay=3, N=100) + noise * jrnd.normal(jrnd.PRNGKey(3), (Ndu,))
y2 = trapz_int(x, G2, u, decay=3, N=100) + noise * jrnd.normal(jrnd.PRNGKey(4), (Ndu,))
# y +=
# jrnd.normal(jrnd.PRNGKey(3), (Ndu,))

plt.scatter(xu, yu)
plt.show()
plt.scatter(x, y1)
plt.scatter(x, y2)
plt.show()
plt.plot(xu, G1(xu))
plt.plot(xu, G2(xu))
plt.show()
# %%

# %%
tg = jnp.linspace(-1, 1, 15)
tu = jnp.linspace(-6, 6, 30).reshape(-1, 1)
model = IOMOVarNVKM(
    [[tg], [tg]],
    tu,
    (xu, yu),
    ([x, x], [y1, y2]),
    q_pars_init=None,
    q_initializer_pars=0.4,
    lsgs=[[0.2], [0.2]],
    ampgs=[[0.7], [0.7]],
    alpha=[1.0, 1.0],
    lsu=0.5,
    ampu=1.0,
    N_basis=50,
    u_noise=noise,
    noise=[noise, noise],
)


# %%
model.plot_samples(xu, [x, x], 2)
# %%
model.plot_filters(jnp.linspace(-2, 2, 500), 3)

# %%
model.fit(500, 2e-3, 30, 20, dont_fit=["noise", "u_noise"])


# %%
tp = jnp.linspace(-6, 6, 150)
model.plot_samples(tp, [tp, tp], 10)
# %%
tf = jnp.linspace(-2, 2, 100)

fig, axs = plt.subplots(2, 1)
g_samps = model.sample_diag_g_gps(
    [[tf.reshape(-1, 1)], [tf.reshape(-1, 1)]], 30, jrnd.split(jrnd.PRNGKey(1), 2)
)
axs[0].plot(
    tf, (g_samps[0][0].T * jnp.exp(-model.alpha[0] * (tf) ** 2)).T, c="red", alpha=0.3,
)
axs[1].plot(
    tf, (g_samps[1][0].T * jnp.exp(-model.alpha[1] * (tf) ** 2)).T, c="red", alpha=0.3,
)
axs[0].plot(tf, G1(tf), label="Truth")
axs[1].plot(tf, G2(tf), label="Truth")
plt.legend()
# %%
model.plot_filters(jnp.linspace(-2, 2, 100), 3)
# %%
