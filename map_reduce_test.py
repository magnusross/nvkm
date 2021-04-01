import jax.numpy as jnp
from jax import vmap
from nvkm.utils import map_reduce, vmap_scan
from nvkm.integrals import integ_1b

ta = jnp.arange(1000).astype(float)
f1 = vmap(lambda t: map_reduce(lambda x, y: t ** (2 / 3) + jnp.sin(x) + y ** 2, ta, ta))


def f(qui, zui):
    print(qui, zui)
    return qui * integ_1b(1.0, 1.0, 1.0, 1, 1.0, zui)


a = jnp.ones(10)
map_reduce(
    f, a, a,
)

g1 = vmap(
    lambda t: map_reduce(lambda x, y: t ** (2 / 3) + jnp.sin(x) + y ** 2, ta, ta)
)(ta)
g2 = vmap_scan(
    lambda t: map_reduce(lambda x, y: t ** (2 / 3) + jnp.sin(x) + y ** 2, ta, ta), ta
)


jnp.all(jnp.isclose(g1, g2))
