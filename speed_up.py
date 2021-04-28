#%%
from nvkm.utils import l2p
from nvkm.models import IOMOVarNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt


zu = jnp.linspace(-1.75, 1.75, 110).reshape(-1, 1)
tg = jnp.linspace(-0.2, 0.2, 10)
tf = jnp.linspace(-0.2, 0.2, 6)
tm2 = jnp.meshgrid(tf, tf)
t2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T

# %%
noise = 0.05
model = IOMOVarNVKM(
    [[tg, t2]],
    zu,
    None,
    None,
    q_pars_init=None,
    q_initializer_pars=0.4,
    lsgs=[[0.05, 0.06]],
    ampgs=[[7.0, 7.0]],
    alpha=[l2p(0.07)],
    lsu=0.04,
    ampu=1.0,
    N_basis=30,
    u_noise=noise,
    noise=[noise],
)
#%%
tp = [jnp.linspace(-2, 2, 30)]
model.sample(tp, 10)
%timeit model.sample(tp, 10)

# %%
