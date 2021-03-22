#%%
from jax import vmap, jit
import jax.numpy as jnp
import jax.random as jrnd
import jax
from nvkm import integrals

#%%
key = jrnd.PRNGKey(10)
c = 3
t = 0.1
zgs = jrnd.uniform(key, shape=(10, 3))
zus = jnp.linspace(-1.0, 1.0, 10)


keys = jrnd.split(key, 8)
thetags = jrnd.normal(keys[0], shape=(15, 3))
betags = jrnd.uniform(keys[1], shape=(15,))

thetus = jrnd.normal(keys[2], shape=(15,))
betaus = jrnd.uniform(keys[3], shape=(15,))

wgs = jrnd.normal(keys[4], shape=(15,))
qgs = jrnd.normal(keys[5], shape=(10,))

wus = jrnd.normal(keys[6], shape=(15,))
qus = jrnd.normal(keys[7], shape=(10,))


#%%
integrals.slow_I(
    t, zgs, zus, thetags, betags, thetus, betaus, wgs, qgs, wus, qus,
)

#%%
def test(a, b):
    return a + b ** 2


def reduceize(f):
    def f_out(s, margs):
        print(margs.shape)
        # print(*margs)
        return s + f(margs)

    return f_out


ta = jnp.arange(6).astype(float)
print(jnp.sum(vmap(lambda x, y: x ** 2 + y ** 2)(ta, ta)))
# print(
#     jax.lax.reduce(
#         (ta, ta), (0.0, 0.0), reduceize(lambda x, y: x ** 2 + y ** 2), [0, 0]
#     )
# )

ft = reduceize(lambda xy: xy[0] ** 2 + xy[1] ** 2)
ft(ft(0.0, jnp.array([2.0, 3.0])), jnp.array([2.0, 3.0]))
jax.lax.reduce(jrnd.uniform(key, shape=(10, 2)), jnp.array([0.0, 0.0]), ft, [0])


#%%
from functools import partial


@partial(jit, static_argnums=(0,))
def map_reduce(f, *arrs):
    sarr = jnp.vstack(arrs).T

    def body_func(i, val):
        return val + f(*sarr[i])

    return jax.lax.fori_loop(0, arrs[0].shape[0], body_func, 0.0)

ta = jnp.arange(10000).astype(float)
%timeit vmap(lambda t: map_reduce(lambda x, y: t ** 2 + x ** 2 + y ** 2, ta, ta))(ta)

#%%
%timeit jnp.sum(vmap(lambda ti: vmap(lambda xi, yi: ti ** 2 + xi ** 2 + yi ** 2)(ta, ta))(ta), axis=1)

# %%
