#%%
from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

keys = jrnd.split(jrnd.PRNGKey(5), 10)
t1 = jnp.linspace(-2.0, 2, 2).reshape(-1, 1)

x = jnp.linspace(-5, 5, 10)
data = (x, jnp.sin(x))
q_pars_init1 = {
    "LC_gs": [jnp.eye(2)],
    "mu_gs": [jnp.sin(t1).flatten()],
    "LC_u": jnp.eye(10),
    "mu_u": jrnd.normal(keys[1], shape=(10,)),
}
var_model1 = VariationalNVKM(
    [t1],
    jnp.linspace(-5, 5, 10).reshape(-1, 1),
    data,
    IndependentGaussians,
    q_pars_init=None,
    q_initializer_pars=0.5,
    lsgs=[1.0],
    ampgs_init=[1.0],
    noise_init=0.01,
    C=1,
)

from datetime import datetime

t1 = datetime.now()
var_model1.compute_bound(10) 
t2 = datetime.now()
print('comp time:', t2-t1)
#%%
 # 96080.31516279
%timeit -n 3 -r 3 var_model1.compute_bound(10).block_until_ready()

# %%
"""
both vmap:
0:00:04.225398
2.8 s ± 647 ms per loop (mean ± std. dev. of 3 runs, 3 loops each)

both scan:
comp time: 0:00:06.074566
1.83 s ± 6.94 ms per loop (mean ± std. dev. of 3 runs, 3 loops each)

scan on sampe
"""
# %%
