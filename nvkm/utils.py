from functools import partial
from typing import Callable, Union, Collection
import operator

from jax.config import config
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrnd
from jax import jit, vmap
import jax
import matplotlib.pyplot as plt


from .settings import JITTER

config.update("jax_enable_x64", True)


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
    op: Callable = operator.add,
):

    # init_val = jnp.zeros_like(f(*[a[0] for a in arrs]))

    # # @jit
    # def body_func(carry, x):
    #     return op(carry, f(*x)), 0.0

    # return jax.lax.scan(body_func, init_val, arrs)[0]
    return jnp.sum(vmap(f)(*arrs))
    # sarr = jnp.vstack((arr.flatten() for arr in arrs)).T
    # # sarr = jnp.vstack(arrs).T

    # def body_func(i, val):
    #     return op(val, f(*sarr[i]))

    # return jax.lax.fori_loop(0, sarr.shape[0], body_func, init_val)


@partial(jit, static_argnums=(0,))
def vmap_scan(f, *arrs):
    def body_func(carry, x):
        return carry, f(*x)

    return jax.lax.scan(body_func, 0.0, arrs)[1]


# def map_reduce_scan(
#     f: Callable,
#     *arrs: jnp.DeviceArray,
#     init_val: float = 0.0,
#     op: Callable = operator.add,
# ):
#     def body_func(carry, x):

#         return carry + f(*x), carry

#     return jax.lax.scan(body_func, 0.0, arrs)[0]


@partial(jit, static_argnums=(0,))
def map_reduce_1vec(
    f: Callable, arr2D: jnp.DeviceArray, *arrs: jnp.DeviceArray,
):
    return jnp.sum(vmap(f)(arr2D, *arrs))
    # sarr = jnp.vstack((arr.flatten() for arr in arrs)).T
    # # sarr = jnp.vstack(arrs).T

    # def body_func(i, val):
    #     return op(val, f(arr2D[i], *sarr[i]))

    # return jax.lax.fori_loop(0, sarr.shape[0], body_func, init_val)
    # init_val = jnp.zeros_like(f(arr2D[0], *[a[0] for a in arrs]))

    # # @jit
    # def body_func(carry, x):
    #     # print(f(*x))
    #     # print(carry)
    #     return carry + f(x[0], *x[1]), 0.0

    # return jax.lax.scan(body_func, init_val, (arr2D, arrs))[0]


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


@jit
def choleskyize(A):
    return jnp.tril(A - 2 * jnp.diag(jnp.diag(A) * (jnp.diag(A) < 0.0)))


def plot_c2_filter_multi(
    model, t, N_s, key=jrnd.PRNGKey(1), save=None, variational=False
):

    tp = jnp.vstack((t, t)).T
    tn = jnp.vstack((t, -t)).T

    if variational:
        skey = jrnd.split(key, N_s + 1)
        gp = model.g_gps[1]

        v_samps = model.q_of_v.sample(model.q_pars, N_s, skey[0])["gs"][1]

        sampsp = vmap(lambda vi, keyi: gp._sample(tp, vi, gp.amp, 1, keyi).flatten())(
            v_samps, skey[1:]
        ).T
        sampsn = vmap(lambda vi, keyi: gp._sample(tn, vi, gp.amp, 1, keyi).flatten())(
            v_samps, skey[1:]
        ).T
    else:
        sampsp = model.g_gps[1].sample(tp, N_s, key=key)
        sampsn = model.g_gps[1].sample(tn, N_s, key=key)
    yp = (sampsp.T * jnp.exp(-model.alpha * (t) ** 2)).T
    yn = (sampsn.T * jnp.exp(-model.alpha * (t) ** 2)).T
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(t, yp, c="red", alpha=0.5)
    axs[0].set_title("Main Diag")
    axs[1].plot(t, yn, c="red", alpha=0.5)
    axs[1].set_title("Off Diag")
    if save:
        plt.savefig(save)
    plt.show()


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


def generate_C2_volterra_data(
    key=jax.random.PRNGKey(345), N_tr=500, N_te=0, C=None, noise=0.5
):
    def k1(x):
        return jnp.exp(-0.2 * x ** 2) * jnp.sin(2 * x)

    def k2(x1, x2):
        return jnp.exp(-(((x1 - 2.0) ** 2 + (x2 - 2.0) ** 2))) - jnp.exp(
            -(((x1 + 2.0) ** 2 + (x2 + 2.0) ** 2))
        )

    nw1 = 1000
    d1 = 30
    u = jax.random.normal(key, (nw1,))
    tau = jnp.linspace(-d1, d1, nw1)
    taux, tauy = jnp.meshgrid(jnp.linspace(-d1, d1, nw1), jnp.linspace(-d1, d1, nw1))

    def int1(t):
        return jnp.sum(k1(t - tau) * u)

    def int2(t):
        return jnp.sum(k2(t - taux, t - tauy) * jnp.outer(u, u)) * 0.1

    N = N_te + N_tr
    x = jnp.linspace(-15, 15, N)
    if C == 1:
        y = jnp.array([int1(xi) for xi in x])
    elif C == 2:
        y = jnp.array([int2(xi) for xi in x])
    else:
        y = jnp.array([int1(xi) + int2(xi) for xi in x])
    # y = jnp.array([ans(xi) for xi in x])
    y = (y - jnp.mean(y)) / jnp.std(y) + noise * jax.random.normal(key, (N,))

    rand_idx = jax.random.permutation(key, jnp.arange(N))
    rand_idx_tr = rand_idx[:N_tr]
    rand_idx_te = rand_idx[N_tr:]

    return x[rand_idx_tr], y[rand_idx_tr], x[rand_idx_te], y[rand_idx_te]


def generate_EQ_data(N=880, key=jax.random.PRNGKey(34)):
    t = jnp.linspace(-44, 44, N)
    K = map2matrix(eq_kernel, t, t, 1.0, 1.0)
    y = jax.random.multivariate_normal(key, jnp.zeros(N), K + 1e-6 * jnp.eye(N))
    noise = 0.3 * jax.random.normal(key, (N,))
    yo = y + noise

    return (
        jnp.concatenate((t[: 440 - 44], t[440 + 44 :])),
        jnp.concatenate((yo[: 440 - 44], yo[440 + 44 :])),
        t,
        y,
    )


# def load_ncmogp_data(
#     N=50, rng=rng, path="/Users/magnus/Documents/phd/code/repos/volterra_gps/data/"
# ):
#     all_x = onp.load(path + "toy_x.npy")
#     all_y = onp.load(path + "toy_y.npy")

#     # shuf = shuffle(rng, np.arange(len(all_x)))
#     # shuf_x = all_x[shuf].T
#     # shuf_y = all_y[shuf].T

#     shuf_tr_x = onp.empty((3, N))
#     shuf_tr_y = onp.empty((3, N))
#     shuf_te_x = onp.empty((3, len(all_x) - N))
#     shuf_te_y = onp.empty((3, len(all_x) - N))

#     for i in range(3):
#         sort_all = np.argsort(all_x[:, i])
#         all_x[:, i] = all_x[sort_all, i]
#         all_y[:, i] = all_y[sort_all, i]
#         mix = shuffle(rng, np.arange(len(all_x)))
#         mixtr = onp.sort(mix[:N])
#         mixte = onp.sort(mix[N:])

#         shuf_tr_x[i] = all_x.T[i, mixtr]
#         shuf_tr_y[i] = all_y.T[i, mixtr]
#         shuf_te_x[i] = all_x.T[i, mixte]
#         shuf_te_y[i] = all_y.T[i, mixte]

#     train = {"X": np.array(shuf_tr_x), "Y": np.array(shuf_tr_y)}
#     test = {"X": np.array(shuf_te_x), "Y": np.array(shuf_te_y)}

#     return train, test
