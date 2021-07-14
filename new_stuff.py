import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap
from nvkm.utils import map2matrix
from nvkm.integrals import Separable, Homogeneous, Full
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
        "t": jnp.linspace(-10, 10, 100),
        "zgs": jrnd.uniform(key, shape=(10,)),
        "zus": jnp.linspace(-1.0, 1.0, 5),
        "thetags": jrnd.normal(keys[0], shape=(11, 3, 50, 1)),
        "betags": jrnd.uniform(keys[1], shape=(11, 3, 50),),
        "thetus": jrnd.normal(keys[2], shape=(11, 50),),
        "betaus": jrnd.uniform(keys[3], shape=(11, 50),),
        "wgs": jrnd.normal(keys[4], shape=(11, 3, 50),),
        "qgs": jrnd.normal(keys[5], shape=(11, 3, 10),),
        "wus": jrnd.normal(keys[6], shape=(11, 50),),
        "qus": jrnd.normal(keys[7], shape=(11, 5),),
        "sigg": 1.0,
    }


data = data_maker()


@partial(
    jnp.vectorize, excluded=(1, 2,), signature="(k)->()",
)
def f(a, b, c):
    return jnp.sum(a * b * c)


# ]


c = Separable.I(
    jnp.linspace(-10, 10, 100),
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
)
