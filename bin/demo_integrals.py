#%%
from jax import vmap, jit
import jax.numpy as jnp
import jax.random as jrnd
import jax
from nvkm import integrals

#%%
key = jrnd.PRNGKey(10)
c = 3
t = 0.1
zgs = jrnd.uniform(key, shape=(10, 3))
zus = jnp.linspace(-1.0, 1.0, 10)


keys = jrnd.split(key, 8)
thetags = jrnd.normal(keys[0], shape=(15, 3))
betags = jrnd.uniform(keys[1], shape=(15,))

thetus = jrnd.normal(keys[2], shape=(10,))
betaus = jrnd.uniform(keys[3], shape=(10,))

wgs = jrnd.normal(keys[4], shape=(15,))
qgs = jrnd.normal(keys[5], shape=(10,))

wus = jrnd.normal(keys[6], shape=(10,))
qus = jrnd.normal(keys[7], shape=(10,))


#%%

#%%
%timeit integrals.slow_I1(t, zus, thetags[0], betags[0], thetus, betaus, wus, qus, 1.0).block_until_ready()
%timeit integrals.fast_I1(t, zus, thetags[0], betags[0], thetus, betaus, wus, qus, 1.0).block_until_ready()
#%%

#%%
%timeit integrals.slow_I2(t, zgs[0], zus, thetus, betaus, wus, qus, 1.0).block_until_ready()
%timeit integrals.fast_I2(t, zgs[0], zus, thetus, betaus, wus, qus, 1.0).block_until_ready()
#%%
#%%

#%%
%timeit integrals.slow_I(t, zgs, zus, thetags, betags, thetus, betaus, wgs, qgs, wus, qus, 1.0).block_until_ready()
#%%
%timeit integrals.fast_I(t, zgs, zus, thetags, betags, thetus, betaus, wgs, qgs, wus, qus, 1.0).block_until_ready()
# %%
