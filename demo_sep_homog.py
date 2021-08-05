#%%
from nvkm.models import (
    SepHomogMOVarNVKM,
    SepMOVarNVKM,
    EQApproxGP,
    MOVarNVKM,
    IOMOVarNVKM,
    SepHomogIOMOVarNVKM,
)
from nvkm.utils import l2p, NMSE, make_zg_grids1D, make_zg_grids, gaussian_NLPD
from nvkm.experiments import load_volterra_data

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap
import pandas as pd
import argparse
from functools import partial
import scipy as osp

import pickle
from pathlib import Path
import os

# %%
keys = jrnd.split(jrnd.PRNGKey(0), 5)

data_dir = "data"


x_train, y_train, x_test, y_test = load_volterra_data(
    0, data_dir=os.path.join(data_dir, "volt")
)

keys = jrnd.split(keys[0], 5)

C = 1
zurange = 1.0
Nvgs = [15] * C
zgran = [0.05] * C
Nvu = 50
zu = jnp.linspace(-zurange, zurange, Nvu).reshape(-1, 1)
lsu = zu[1][0] - zu[0][0]
zgs, lsgs = make_zg_grids1D(zgran, Nvgs, causal=False)


model = SepHomogMOVarNVKM(
    [zgs],
    zu,
    ([None], [None]),
    q_frac=0.00001,
    lsgs=[lsgs],
    ampgs=[[1.0] * C],
    key=keys[0],
    noise=[0.05],
    alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]],
    lsu=lsu,
    ampu=1.0,
    N_basis=20,
    causal=False,
)

#%%
model.fit(
    200,
    1e-3,
    50,
    10,
    dont_fit=["lsgs", "lsu", "noise"],
    key=keys[1],
)

#%%
model.plot_samples(
    jnp.linspace(-zurange, zurange, 300), [jnp.linspace(-zurange, zurange, 300)], 10
)
#%%
model.plot_filters(jnp.linspace(-zgran[0], zgran[0], 100), 15)

# %%
m, s = model.predict([x_test], 20)

test_nmse = NMSE(m[0], y_test)
test_nlpd = gaussian_NLPD(m[0], s[0], y_test)
print("Test NMSE:", test_nmse)
print("Test NLPD:", test_nlpd)
# %%
