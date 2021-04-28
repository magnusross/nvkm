#%%
import argparse
from nvkm.models import IOMOVarNVKM
from jax.config import config
import numpy as onp
from scipy.integrate import quad, dblquad

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
from functools import partial
from nvkm.utils import RMSE

parser = argparse.ArgumentParser(description="Compare CPU and GPU times.")
parser.add_argument("--Nits", default=10, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--f_name", default="test", type=str)
args = parser.parse_args()
#%%
def u(x):
    return (
        jnp.cos(0.3 * x)
        + jnp.sin(2 * x) * jnp.cos(2 * x) ** 2
        - jnp.cos(0.4 * x)
        + jnp.cos(4 * x) * jnp.sin(1 * x) ** 2
    )


def G1(x, a=3.0, b=0.1, alpha=0.5):
    return jnp.exp(-alpha * x ** 2) * (-onp.sin(a * x) + b - b * x ** 2)


def G2(x, a=1.5, b=0.1, alpha=0.5):
    return jnp.exp(-alpha * x ** 2) * (-onp.cos(a * x) ** 2 + (x + b) ** 2)


def trapz_int(t, h, x, dim=1, decay=4, N=100):
    tau = jnp.linspace(t - decay, t + decay, N)
    ht = h(t - tau)
    xt = x(tau)
    return jnp.trapz(ht * xt, x=tau, axis=0)


Ndu = 500
xu = jnp.linspace(-15, 15, Ndu)

fyc1 = lambda x: trapz_int(x, G1, u, decay=4.0, N=300)
yc1 = fyc1(xu)
fyc2 = lambda x: trapz_int(
    x, G2, lambda xui: trapz_int(xui, G2, u, decay=4.0, N=300), decay=4.0, N=300
)
yc2 = fyc2(xu)
noise = 0.1
yu = u(xu) + noise * jrnd.normal(jrnd.PRNGKey(2), (Ndu,))
y = yc1 + yc2 + noise * jrnd.normal(jrnd.PRNGKey(4), (Ndu,))


# fig = plt.figure(figsize=(30, 3))
# plt.plot(xu, yu)
# plt.plot(xu, y)
# plt.show()

# tg = jnp.linspace(-3, 3, 100)
# plt.plot(tg, G1(tg))
# plt.plot(tg, G2(tg) * G2(tg))
# plt.show()

N_train = 300
x = xu
train_data = ([x[:N_train]], [y[:N_train]])
test_data = ([x[N_train:]], [y[N_train:]])

tg = jnp.linspace(-2, 2, 25)
tf = jnp.linspace(-3, 3, 5)
tm2 = jnp.meshgrid(tf, tf)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T
tu = jnp.linspace(-16, 16, 70).reshape(-1, 1)
#%%
modelc1 = IOMOVarNVKM(
    [[tg]],
    tu,
    (xu, yu),
    train_data,
    q_pars_init=None,
    q_initializer_pars=0.4,
    lsgs=[[0.4]],
    ampgs=[[0.7]],
    alpha=[0.5],
    lsu=0.5,
    ampu=1.0,
    N_basis=50,
    u_noise=noise,
    noise=[noise],
)


modelc2 = IOMOVarNVKM(
    [[tg, t2]],
    tu,
    (xu, yu),
    train_data,
    q_pars_init=None,
    q_initializer_pars=0.4,
    lsgs=[[0.6, 0.6]],
    ampgs=[[0.7, 0.7]],
    alpha=[0.5],
    lsu=0.5,
    ampu=1.0,
    N_basis=50,
    u_noise=noise,
    noise=[noise],
)
#%%

modelc1.fit(args.Nits, args.lr, 30, 10, dont_fit=["noise", "u_noise"])
modelc1.save(args.f_name + "io_c1_model.pkl")

modelc2.fit(args.Nits, args.lr, 30, 10, dont_fit=["noise", "u_noise"])
modelc2.save(args.f_name + "io_c2_model.pkl")
#%%
tp = jnp.linspace(-15, 15, 200)
axs = modelc1.plot_samples(tp, [tp], 10, return_axs=True)
axs[0].plot(tp, u(tp), c="black", ls=":", label="True $u$")
axs[1].plot(tp, fyc1(tp) + fyc2(tp), c="black", ls=":", label="True $y_1$")
plt.savefig(args.f_name + "c1samps.png", dpi=800)
plt.show()

tf = jnp.linspace(-2.5, 2.5, 100)
axs = modelc1.plot_filters(tf, 3, return_axs=True)
plt.savefig(args.f_name + "c1filters.png", dpi=800)
plt.show()

tp = jnp.linspace(-15, 15, 200)
axs = modelc2.plot_samples(tp, [tp], 10, return_axs=True)
axs[0].plot(tp, u(tp), c="black", ls=":", label="True $u$")
axs[1].plot(tp, fyc1(tp) + fyc2(tp), c="black", ls=":", label="True $y_1$")
plt.savefig(args.f_name + "c2samps.png", dpi=800)
plt.show()

tf = jnp.linspace(-2.5, 2.5, 100)
axs = modelc2.plot_filters(tf, 3, return_axs=True)
axs[0].plot(
    tf, G1(tf), label="Truth", c="black", ls=":",
)
axs[1].plot(
    tf, G2(tf) * G2(tf), label="Truth", c="black", ls=":",
)
plt.savefig(args.f_name + "c2filters.png", dpi=800)
plt.show()

# %%
train_preds1 = modelc1.sample(train_data[0], 300)
train_preds2 = modelc2.sample(train_data[0], 300)

test_preds1 = modelc1.sample(test_data[0], 300)
test_preds2 = modelc2.sample(test_data[0], 300)
#%%
print("c1 train RMSE:", RMSE(train_data[1][0], jnp.mean(train_preds1[0], axis=1)))
print("c2 train RMSE:", RMSE(train_data[1][0], jnp.mean(train_preds2[0], axis=1)))

print("c1 test RMSE:", RMSE(test_data[1][0], jnp.mean(test_preds1[0], axis=1)))
print("c2 test RMSE:", RMSE(test_data[1][0], jnp.mean(test_preds2[0], axis=1)))
# %%