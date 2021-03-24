#%%
from nvkm.models import NVKM
import jax.numpy as jnp
import jax.random as jrnd

keys = jrnd.split(jrnd.PRNGKey(5), 10)

t1 = jnp.linspace(-2.0, 2, 10).reshape(-1, 1)
t2 = 2 * jrnd.uniform(keys[0], shape=(5, 2)) - 1.0
t3 = 2 * jrnd.uniform(keys[0], shape=(5, 3)) - 1.0
b = NVKM(
    zu=jnp.linspace(-10, 10, 20).reshape(-1, 1),
    vu=jrnd.normal(keys[1], shape=(20,)),
    zgs=[t1, t2, t3],
    vgs=[jnp.sin(t1).flatten(), jnp.sin(t2[:, 0] ** 2), jnp.sin(t3[:, 0] ** 2)],
    lsgs=[1.0, 2.0, 0.5],
    ampgs=[1.0, 2.0, 0.5],
    C=3,
)
#%%


# %%
import matplotlib.pyplot as plt

t = jnp.linspace(-10, 10, 100)
test = b.sample(t, N_s=2)
plt.plot(t, test)

# %%
t = jnp.linspace(-10, 10, 20)
%timeit b.sample(t, N_s=3)
# %%
