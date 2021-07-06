import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap
from nvkm.utils import map2matrix
from nvkm.integrals import Separable


@jit
def f(a, b, c, d):
    return a * c * b * d


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
        "zgs": jrnd.uniform(key, shape=(5, 3)),
        "zus": jnp.linspace(-1.0, 1.0, 5),
        "thetags": jrnd.normal(keys[0], shape=(8, 3)),
        "betags": jrnd.uniform(keys[1], shape=(8,)),
        "thetus": jrnd.normal(keys[2], shape=(5,)),
        "betaus": jrnd.uniform(keys[3], shape=(5,)),
        "wgs": jrnd.normal(keys[4], shape=(8,)),
        "qgs": jrnd.normal(keys[5], shape=(5,)),
        "wus": jrnd.normal(keys[6], shape=(5,)),
        "qus": jrnd.normal(keys[7], shape=(5,)),
        "sigg": 1.0,
    }


data = data_maker()

# %timeit map2matrix(lambda t1, t2: f(*t1, *t2), (a, a), (a, a))
# %timeit map2matrix(g, a, a)
# %timeit test(a)

%timeit Separable.single_I(data["t"],data["zgs"],data["zus"],data["thetags"],data["betags"],data["thetus"],data["betaus"],data["wgs"],data["qgs"],data["wus"],data["qus"],data["sigg"],1.1,1.2,1.3,1.4,)
