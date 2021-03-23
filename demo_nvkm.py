from nvkm.models import NVKM
import jax.numpy as jnp
import jax.random as jrnd

keys = jrnd.split(jrnd.PRNGKey(5), 10)

t1 = jnp.linspace(-2.0, 2, 10).reshape(-1, 1)
t2 = 2 * jrnd.uniform(keys[0], shape=(5, 2)) - 1.0
b = NVKM(
    zu=jnp.linspace(-10, 10, 20).reshape(-1, 1),
    vu=jrnd.normal(keys[1], shape=(20,)),
    zgs=[t1, t2],
    vgs=[jnp.sin(t1).flatten(), jnp.sin(t2[:, 0] ** 2)],
    lsgs=[1.0, 2.0],
    ampgs=[1.0, 2.0],
    C=2,
)

b.sample(jnp.linspace(-10, 10, 100))
