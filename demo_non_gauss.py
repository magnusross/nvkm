#%%
from nvkm.utils import generate_C2_volterra_data, l2p
from nvkm.models import NVKM, VariationalNVKM
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

#%%
t1 = jnp.linspace(-1, 1, 20)
t2 = 2.0 * jrnd.uniform(jrnd.PRNGKey(1), (20, 2)) - 1
t3 = 2.0 * jrnd.uniform(jrnd.PRNGKey(1), (10, 3)) - 1

model = NVKM(
    zgs=[t1, t2, t3],
    vgs=[None, None, None],
    zu=jnp.linspace(-10, 10, 30).reshape(-1, 1),
    vu=None,
    C=3,
    lsgs=[0.4, 0.4, 0.4],
    ampgs=[1.0, 1.0, 1.0],
    alpha=l2p(0.4),
    N_basis=100,
)

t = jnp.linspace(-10, 10, 200)
s = model.sample(t, 10)
plt.plot(t, s)
plt.show()

# %%
sp = model.sample(jnp.array([-2.5, 7.5]), 1000)
# %%
plt.scatter(sp[0], sp[1], alpha=0.3)
# %%
ss = jnp.array([])
for i in range(10):
    ss = jnp.append(ss, model.sample(jnp.array([2.5]), 1000))

# %%
plt.hist(ss, bins=50)

plt.show()
# %%
