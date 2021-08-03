import operator
import pickle
from functools import partial
from typing import Callable, Collection, Union

from jax.config import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as onp
from jax import jit, vmap
from tensorflow_probability.substrates import jax as tfp


from .settings import JITTER


@partial(jit, static_argnums=(0,))
def map2matrix(f: Callable, ts: jnp.ndarray, tps: jnp.ndarray, *args) -> jnp.ndarray:
    """
    Uses Jax's vectorised map, to apply f to app pairs of elements
    in t, and tp. Most often uses to build covainces matices,
    where f is a kernel, and ts == tps, are the inputs. The first
    2 arguments of f are mapped, extra non mapped argumnents for
    f go in args.

    Args:
        f (Callable): Function to be mapped.
        ts (jnp.ndarray): First array to map
        tps (jnp.ndarray): Second array to map

    Returns:
        jnp.ndarray: Ouput matrix.
    """
    return vmap(lambda ti: vmap(lambda tpi: f(ti, tpi, *args))(tps))(ts)


@partial(jit, static_argnums=(0,))
def map_reduce(
    f: Callable,
    *arrs: jnp.ndarray,
    init_val: float = 0.0,
    op: Callable = operator.add,
):
    """
    helper function for performing a map then a sum.
    """
    return jnp.sum(vmap(f)(*arrs))


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
    """
    Root mean square error
    """
    return jnp.sqrt((jnp.sum((yp - ye) ** 2)) / len(yp))


@jit
def NMSE(yp, ytrue):
    """
    Normalised mean square error
    """
    return jnp.mean((yp - ytrue) ** 2) / jnp.mean((jnp.mean(ytrue) - ytrue) ** 2)


@jit
def gaussian_NLPD(yp, ypvar, ytrue):
    """
    Gaussian negative log probability density.
    """
    return jnp.mean(0.5 * jnp.log(2 * jnp.pi * ypvar) + (ytrue - yp) ** 2 / (2 * ypvar))


@jit
def eq_kernel(
    t: Union[jnp.ndarray, float],
    tp: Union[jnp.ndarray, float],
    amp: float,
    ls: float,
) -> float:
    """
    EQ kernel in 1D if inputs are float, if inputs are array then
    the it is the isotropic EQ kerenl in len(t) dimesions.

    Args:
        t (Union[jnp.ndarray, float]): First input time.
        tp (Union[jnp.ndarray, float]): Second input time.
        amp (float): Amplitude.
        ls (float): Length scale.

    Returns:
        float: Covariance between points t and tp.
    """
    return amp ** 2 * jnp.exp(-0.5 * jnp.sum((t - tp) ** 2) / ls ** 2)


@jit
def choleskyize(A):
    """
    Enforces array as vaild cholesky decompostion, for optimisizing covariance
    matrices.
    """
    return jnp.tril(A - 2 * jnp.diag(jnp.diag(A) * (jnp.diag(A) < 0.0)))


def exact_gp_posterior(kf, ts, zs, us, *kf_args, noise=0.0, jitter=JITTER):
    """
    Gives mean and covarince of exact GP.
    """
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


def make_zg_grids(zgran: list, Nvgs: list):
    """
    Lays out points on grid for each order VK.
    """
    tgs = []
    lsgs = []
    for i in range(len(Nvgs)):
        tg = jnp.linspace(-zgran[i], zgran[i], Nvgs[i])
        lsgs.append(1.5 * (tg[1] - tg[0]))
        tm2 = jnp.meshgrid(*[tg] * (i + 1))
        tgs.append(jnp.vstack([tm2[k].flatten() for k in range(i + 1)]).T)
    return tgs, lsgs


def make_zg_grids1D(zgran: list, Nvgs: list):
    tgs = []
    lsgs = []
    for i in range(len(Nvgs)):
        tg = jnp.linspace(-zgran[i], zgran[i], Nvgs[i]).reshape(-1, 1)
        lsgs.append((tg[1][0] - tg[0][0]))
        tgs.append(tg)
    return tgs, lsgs


@jit
def erfi(x):
    """Approximation of the imaginary error function, valid for real arguments."""
    return 2.0 * tfp.math.dawsn(x) * jnp.exp(jnp.square(x)) / jnp.sqrt(jnp.pi)


@jit
def l2norm(x1, x2):
    return jnp.sqrt(jnp.square(x1) + jnp.square(x2))
