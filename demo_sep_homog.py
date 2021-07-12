#%%
from nvkm.models import SepHomogMOVarNVKM, EQApproxGP, MOVarNVKM
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


#%%
keys = jrnd.split(jrnd.PRNGKey(0), 5)

data_dir = "data"


x_train, y_train, x_test, y_test = load_volterra_data(
    0, data_dir=os.path.join(data_dir, "volt")
)

keys = jrnd.split(keys[0], 5)

zurange = 22.0
Nvgs = [15, 8]
zgran = [1.5] * 2
Nvu = 88
C = len(Nvgs)
zu = jnp.linspace(-zurange, zurange, Nvu).reshape(-1, 1)
lsu = zu[1][0] - zu[0][0]
zgs, lsgs = make_zg_grids(zgran, Nvgs)


model = MOVarNVKM(
    [zgs],
    zu,
    ([x_train], [y_train]),
    q_frac=0.8,
    lsgs=[lsgs],
    ampgs=[[0.8] * 2],
    key=keys[0],
    noise=[0.05],
    alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]],
    lsu=lsu,
    ampu=1.0,
    N_basis=20,
)

#%%
model.fit(
    500, 5e-3, 50, 10, dont_fit=["lsgs", "lsu", "noise"], key=keys[1],
)
#%%

model.plot_samples(jnp.linspace(-22, 22, 300), [jnp.linspace(-22, 22, 300)], 10)
model.plot_filters(jnp.linspace(-1.5, 1.5, 100), 15)


# %%

m, s = model.predict([x_test], 50)

test_nmse = NMSE(m[0], y_test)
test_nlpd = gaussian_NLPD(m[0], s[0], y_test)
print("Test NMSE:", test_nmse)
print("Test NLPD:", test_nlpd)

# %%
