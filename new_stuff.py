import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap
from nvkm.utils import map2matrix
from nvkm.integrals import Separable
from functools import partial


@jit
def g(a, b):
    return a * a * b * a


@jit
def test(ar):
    return vmap(lambda a, b: vmap(lambda c, d: f(a, b, c, d))(ar, ar))(ar, ar)


def data_maker():
    key = jrnd.PRNGKey(10)
    keys = jrnd.split(key, 8)
    return {
        "t": 0.1,
        "zgs": jrnd.uniform(key, shape=(8,)),
        "zus": jnp.linspace(-1.0, 1.0, 5),
        "thetags": jrnd.normal(keys[0], shape=(10, 2, 8)),
        "betags": jrnd.uniform(
            keys[1],
            shape=(
                10,
                2,
                8,
            ),
        ),
        "thetus": jrnd.normal(
            keys[2],
            shape=(
                10,
                5,
            ),
        ),
        "betaus": jrnd.uniform(
            keys[3],
            shape=(
                10,
                5,
            ),
        ),
        "wgs": jrnd.normal(
            keys[4],
            shape=(
                10,
                2,
                8,
            ),
        ),
        "qgs": jrnd.normal(
            keys[5],
            shape=(
                10,
                2,
                8,
            ),
        ),
        "wus": jrnd.normal(
            keys[6],
            shape=(
                10,
                5,
            ),
        ),
        "qus": jrnd.normal(
            keys[7],
            shape=(
                10,
                5,
            ),
        ),
        "sigg": 1.0,
    }


data = data_maker()


@partial(
    jnp.vectorize,
    excluded=(
        1,
        2,
    ),
    signature="(k)->()",
)
def f(a, b, c):
    return jnp.sum(a * b * c)


# vmap(f)(jnp.ones((10, 3, 10)), jnp.ones((10, 10)), jnp.ones((10, 10)))
# %timeit map2matrix(lambda t1, t2: f(*t1, *t2), (a, a), (a, a))
# %timeit map2matrix(g, a, a)
# %timeit test(a)

# %timeit Separable.single_I(0.1,data["zgs"],data["zus"],data["thetags"],data["betags"],data["thetus"],data["betaus"],data["wgs"],data["qgs"],data["wus"],data["qus"],data["sigg"],1.1,1.2,1.3,1.4,)
# %timeit vmap(jit(lambda a: vmap(jit(lambda b: jnp.cos(b)*b**10 / 2))(a)))(jnp.ones((10, 10)))
# %timeit vmap(lambda a: vmap(lambda b: a*jnp.cos(b)*b**10 / 2)(a))(jnp.ones((1000, 1000, 10)))
# %timeit map2matrix(lambda a, b: jnp.cos(b)*b**10 / 2)

# jnp.vectorize(lambda a, b: a * jnp.dot(b, b), signature="(k)->()", excluded=(0,))(
#     10.0, jnp.ones((100, 100, 10))
# )
Separable.I(
    jnp.ones(100),
    data["zgs"],
    data["zus"],
    data["thetags"],
    data["betags"],
    data["thetus"],
    data["betaus"],
    data["wgs"],
    data["qgs"],
    data["wus"],
    data["qus"],
    data["sigg"],
    1.1,
    1.2,
).shape
