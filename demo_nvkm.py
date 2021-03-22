from nvkm.models import NVKM
import jax.numpy as jnp

t = jnp.linspace(-1.0, 1, 10)
a = NVKM()
b = NVKM(
    zgs=[jnp.ones(5), jnp.ones((5, 2)), None],
    vgs=[jnp.ones(5), jnp.ones(5), None],
    lsgs=[1.0, 2.0, 3.0],
    ampgs=[1.0, 2.0, 3.0],
    C=3,
)

