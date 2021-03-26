#%%
from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

keys = jrnd.split(jrnd.PRNGKey(5), 10)

t1 = jnp.linspace(-2.0, 2, 2).reshape(-1, 1)
t2 = 2 * jrnd.uniform(keys[0], shape=(2, 2)) - 1.0
t3 = 2 * jrnd.uniform(keys[0], shape=(2, 3)) - 1.0

x = jnp.linspace(-5, 5, 1)
data = (x, jnp.sin(x))
q_pars_init3 = {
    "LC_gs": [jnp.eye(2) for i in range(3)],
    "mu_gs": [jnp.sin(t1).flatten(), jnp.sin(t2[:, 0] ** 2), jnp.sin(t3[:, 0] ** 2)],
    "LC_u": jnp.eye(2),
    "mu_u": jrnd.normal(keys[1], shape=(2,)),
}
var_model3 = VariationalNVKM(
    [t1, t2, t3],
    jnp.linspace(-5, 5, 2).reshape(-1, 1),
    data,
    IndependentGaussians,
    q_pars_init=q_pars_init3,
    lsgs=[1.0, 2.0, 1.0],
    ampgs_init=[1.0, 1.0, 1.0],
    noise_init=0.01,
    C=3,
)

print(
    var_model3._compute_bound(data, var_model3.q_of_v.q_pars, var_model3.ampgs, 0.1, 2)
)
var_model3.fit(100, 1e-1, 1, 2)

# %%
t = jnp.linspace(-5, 5, 100)
samps = var_model3._var_sample(t, var_model3.q_of_v.q_pars, var_model3.ampgs, 2)
plt.plot(t, samps)


# %%
