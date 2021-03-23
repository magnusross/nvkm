from functools import partial
from typing import Callable, Union, Collection
import operator

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
import jax

from .settings import JITTER


@partial(jit, static_argnums=(0,))
def map2matrix(
    f: Callable, ts: jnp.DeviceArray, tps: jnp.DeviceArray, *args
) -> jnp.DeviceArray:
    """
    Uses Jax's vectorised map, to apply f to app pairs of elements
    in t, and tp. Most often uses to build covainces matices,
    where f is a kernel, and ts == tps, are the inputs. The first
    2 arguments of f are mapped, extra non mapped argumnents for
    f go in args.

    Args:
        f (Callable): Function to be mapped.
        ts (jnp.DeviceArray): First array to map
        tps (jnp.DeviceArray): Second array to map

    Returns:
        jnp.DeviceArray: Ouput matrix.
    """
    return vmap(lambda ti: vmap(lambda tpi: f(ti, tpi, *args))(tps))(ts)


@partial(jit, static_argnums=(0,))
def map_reduce(
    f: Callable,
    *arrs: jnp.DeviceArray,
    init_val: float = 0.0,
    op: Callable = operator.add
):
    sarr = jnp.vstack(arrs).T

    def body_func(i, val):
        return op(val, f(*sarr[i]))

    return jax.lax.fori_loop(0, sarr.shape[0], body_func, init_val)


def map_reduce_1vec(
    f: Callable,
    arr2D: jnp.DeviceArray,
    *arrs: jnp.DeviceArray,
    init_val: float = 0.0,
    op: Callable = operator.add
):

    sarr = jnp.vstack(arrs).T
    # print(arr2D.shape)

    def body_func(i, val):
        return op(val, f(arr2D[i], *sarr[i]))

    return jax.lax.fori_loop(0, sarr.shape[0], body_func, init_val)


@jit
def l2p(l: float):
    """
    lengthscale to prescision
    """
    return 0.5 * (1 / l ** 2)


@jit
def p2l(p: float):
    """
    precision to lengthscale
    """
    return 1 / jnp.sqrt(2 * p)


@jit
def RMSE(yp, ye):
    return jnp.sqrt((jnp.sum((yp - ye) ** 2)) / len(yp))


@jit
def eq_kernel(
    t: Union[jnp.DeviceArray, float],
    tp: Union[jnp.DeviceArray, float],
    amp: float,
    ls: float,
) -> float:
    """
    EQ kernel in 1D if inputs are float, if inputs are array then
    the it is the isotropic EQ kerenl in len(t) dimesions.

    Args:
        t (Union[jnp.DeviceArray, float]): First input time.
        tp (Union[jnp.DeviceArray, float]): Second input time.
        amp (float): Amplitude.
        ls (float): Length scale.

    Returns:
        float: Covariance between points t and tp.
    """
    return amp ** 2 * jnp.exp(-0.5 * jnp.sum((t - tp) ** 2) / ls ** 2)


def method(cls):
    """Decorator to add the function as a method to a class.
    Args:
        cls (type): Class to add the function as a method to.
    """

    def decorator(f):
        setattr(cls, f.__name__, f)
        return f

    return decorator


def exact_gp_posterior(kf, ts, zs, us, *kf_args, noise=0.0, jitter=JITTER):

    Koo = map2matrix(kf, zs, zs, *kf_args) + (noise + JITTER) * jnp.eye(len(zs))
    Kop = map2matrix(kf, zs, ts, *kf_args)
    Kpp = map2matrix(kf, ts, ts, *kf_args)

    Loo = jsp.linalg.cholesky(Koo, lower=True)
    Kinvy = jsp.linalg.solve_triangular(
        Loo.T, jsp.linalg.solve_triangular(Loo, us, lower=True)
    )
    Lop = jsp.linalg.solve_triangular(Loo, Kop, lower=True)

    m_post = jnp.dot(Kop.T, Kinvy)
    K_post = Kpp - jnp.dot(Lop.T, Lop)

    return m_post, K_post
