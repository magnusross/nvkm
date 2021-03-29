#%%
from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

keys = jrnd.split(jrnd.PRNGKey(5), 10)
t1 = jnp.linspace(-2.0, 2, 2).reshape(-1, 1)

x = jnp.linspace(-5, 5, 5)
data = (x, jnp.sin(x))
q_pars_init1 = {
    "LC_gs": [jnp.eye(2)],
    "mu_gs": [jnp.sin(t1).flatten()],
    "LC_u": jnp.eye(5),
    "mu_u": jrnd.normal(keys[1], shape=(5,)),
}
var_model1 = VariationalNVKM(
    [t1],
    jnp.linspace(-5, 5, 5).reshape(-1, 1),
    data,
    IndependentGaussians,
    q_pars_init=q_pars_init1,
    lsgs=[1.0],
    ampgs_init=[1.0],
    noise_init=0.01,
    C=1,
)


print(var_model1.compute_bound(2))
print(var_model1.q_pars)
# %%
t = jnp.linspace(-10, 10, 100)
samps = var_model1.sample(t, 2)
#%%
plt.plot(t, samps)
plt.scatter(data[0], data[1])
plt.show()
# %%
var_model1.fit(100, 1e-2, None, 5)
print(var_model1.compute_bound(2))
print(var_model1.noise)
t = jnp.linspace(-5, 5, 100)

#%%
samps = var_model1.sample(t, 5, key=jrnd.PRNGKey(12))
plt.plot(t, samps)
plt.scatter(data[0], data[1])
plt.show()

# %%
var_model1.plot_samples(t, 10)


# %%
var_model1.plot_filters(jnp.linspace(-3, 3, 100), 30)

# %%
