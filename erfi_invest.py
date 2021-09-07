from jax.scipy.special import erf, gammainc
import jax.numpy as jnp
from jax import lax, jit
from scipy.special import erfi


# import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow_probability.substrates import jax as tfp

# %timeit erfi(0.5)

t = jnp.linspace(-5, 5, 100)
# plt.plot(t, erfi(t))
# plt.show()

# # @jit
# # def jerfi(x):
# # return erfi(x)
# # tf.math.special.dawsn(t)
# # plt.plot(t, tf.math.special.dawsn(t))

# jit(tfp.math.dawsn)(t)
# # tfp.


@jit
def jerfi(x):
    return 2.0 * tfp.math.dawsn(x) * jnp.exp(jnp.square(x)) / jnp.sqrt(jnp.pi)


plt.plot(t, jerfi(t))
plt.show()
plt.plot(t, jerfi(t) - erfi(t))
