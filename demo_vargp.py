#%%
from nvkm.models import SepHomogMOVarNVKM, EQApproxGP, VarEQApproxGP
from nvkm.utils import l2p, NMSE, make_zg_grids1D, gaussian_NLPD
from nvkm.vi import gaussian_likelihood
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap, grad
import pandas as pd
import argparse
from functools import partial
import scipy as osp

import pickle
from pathlib import Path
import os

#%%
keys = jrnd.split(jrnd.PRNGKey(0), 5)

zu = jnp.linspace(-10, 10, 30).reshape(-1, 1)

model = VarEQApproxGP(zu, q_frac=1.0)


# %%

# %%


# %%
s = model._sample(model.z, model.mu, model.LC, 1.0, 1.0, 10, keys[0])
# %%
plt.plot(model.z, s)
# %%
params = {"mu": model.mu, "LC": model.LC, "amp": 1.0, "ls": 1.0, "noise": 0.1}

x = jnp.linspace(-10, 10, 500)
y = jnp.sin(x) + 0.2 * jrnd.normal(keys[1], (500,))


@jit
def loss(params, y, key, Ns=10):
    s = model._sample(
        x.reshape(-1, 1),
        params["mu"],
        params["LC"],
        params["amp"],
        params["ls"],
        Ns,
        key,
    )
    like = gaussian_likelihood(y, s, 0.01)
    _, LKvv = model.compute_covariances(params["amp"], params["ls"])
    return -(like + model._KL(params["LC"], params["mu"], LKvv))


# %%
loss(params, y, keys[1])
# %%
step_size = 0.000001


@jit
def update(params, y, key):
    grads = grad(loss)(params, y, key)
    out = {}
    for k in params.keys():
        out[k] = params[k] - grads[k] * step_size
    return out


for i in range(1000):
    keys = jrnd.split(keys[0], 2)
    params = update(params, y, keys[0])
    # print(params)

    if i % 100 == 0:
        print(loss(params, y, keys[0]))

# %%
fitted = VarEQApproxGP(zu, **params,)
# %%
s = fitted.sample(x.reshape(-1, 1), 10, keys[0])
# %%
fig = plt.figure(figsize=(10, 2))
plt.plot(x, s, c="green", alpha=0.5)
plt.plot(x, jnp.sin(x), ls=":", c="red")
plt.scatter(x, y, alpha=0.5)
plt.scatter(zu, params["mu"])
plt.show()
# %%
params
# %%
