#%%
from nvkm.utils import l2p
from nvkm.models import IOMOVarNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import pandas as pd
import argparse

#%%
data = pd.read_csv("data/silverbox.csv")
data = data.drop(columns=["Unnamed: 2"])
# %%
fig = plt.figure(figsize=(30, 2))
plt.plot(data[9000:10000])
plt.vlines(jnp.linspace(9000, 10000, 200), -0.05, 0.05)

# %%
