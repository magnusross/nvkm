from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap
import pandas as pd
import matplotlib.pyplot as plt
import scipy as osp
import copy
from functools import partial
import os
from scipy.io import loadmat
from .models import EQApproxGP


def generate_volterra_data(path: str = os.path.join("data", "volt"),):
    """
    Generates synthetic data. 

    Args:
        path (str, optional): Where to save. Defaults to os.path.join("data", "volt").
    """
    key1, key2 = jrnd.split(jrnd.PRNGKey(0), 2)
    for rep in range(0, 10):

        #%%
        gp1D = EQApproxGP(
            z=None, v=None, amp=1.0, ls=0.5, noise=0.0001, N_basis=50, D=1
        )

        @jit
        def gp_forcing(t):
            return gp1D.sample(t, 1, key=key1).flatten()

        @jit
        def G1(x, a=1.0, b=1.0, alpha=2):
            return jnp.exp(-alpha * x ** 2) * (-jnp.sin(6 * x))

        @jit
        def G2(x, a=1.0, b=1.0, alpha=2):
            return jnp.exp(-alpha * x ** 2) * (jnp.sin(5 * x) ** 2)

        @jit
        def G3(x, a=1.0, b=1.0, alpha=2):
            return jnp.exp(-alpha * x ** 2) * (jnp.cos(-4 * x))

        @partial(jit, static_argnums=(1, 2, 3))
        def trapz_int(t, h, x, N, dim=1, decay=4):
            tau = jnp.linspace(t - decay, t + decay, N)
            ht = h(t - tau)
            xt = x(tau)
            return jnp.trapz(ht * xt, x=tau, axis=0)

        N = 1200
        t = jnp.linspace(-20, 20, N)
        Nint = 100

        fyc1 = jit(lambda x: trapz_int(x, G1, gp_forcing, Nint, decay=3.0))
        fyc2 = jit(lambda x: trapz_int(x, G2, gp_forcing, Nint, decay=3.0))
        fyc3 = jit(lambda x: trapz_int(x, G3, gp_forcing, Nint, decay=3.0))
        #%%
        yc1 = vmap(fyc1)(t)
        yc2 = vmap(fyc2)(t)
        yc3 = vmap(fyc3)(t)
        # yc3 = vmap(fyc3)(t) ** 3

        y = 5 * yc1 * yc2 + 5 * yc3 ** 3
        y = jnp.minimum(y, 1 * jnp.ones_like(y)) + 0.05 * jrnd.normal(key2, (N,))
        _, key2 = jrnd.split(key2)
        # %%
        # plt.plot(tg, G3(tg) * G3(tg) * G3(tg))
        #%%
        Ntr = 400
        all_idx = jnp.arange(N)
        tridx = jrnd.choice(key2, all_idx, (Ntr,), replace=False)

        teidx = jnp.setdiff1d(all_idx, tridx)
        x_train, y_train = t[tridx], y[tridx]

        x_test, y_test = t[teidx], y[teidx]
        pd.DataFrame({"x_train": x_train, "y_train": y_train}).to_csv(
            os.path.join(path, "rep" + str(rep) + "train.csv")
        )
        pd.DataFrame({"x_test": x_test, "y_test": y_test}).to_csv(
            os.path.join(path, "rep" + str(rep) + "test.csv")
        )
        print(rep, "done")
        _, key2 = jrnd.split(key2)


def load_volterra_data(rep: int, data_dir: str = "data") -> tuple:
    """
    Loads sythetic data. 

    Args:
        rep (int): The repeat number.
        data_dir (str, optional): data directory Defaults to "data".

    Returns:
        tuple: data in form (x_train, y_train, x_test, y_test)
    """
    path = os.path.join(data_dir, "rep" + str(rep))
    tr_df = pd.read_csv(path + "train.csv")
    te_df = pd.read_csv(path + "test.csv")
    return (
        jnp.array(tr_df["x_train"]),
        jnp.array(tr_df["y_train"]),
        jnp.array(te_df["x_test"]),
        jnp.array(te_df["y_test"]),
    )


class MODataSet:
    """
    Generic class for multiouput datasets.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_data()
        self.O = len(self.train_x)
        self.compute_scales()
        self.strain_x, self.strain_y = self.downscale(self.train_x, self.train_y)
        self.stest_x, self.stest_y = self.downscale(self.test_x, self.test_y)

    def load_data(self):
        self.train_x = [None]
        self.train_y = [None]
        self.test_x = [None]
        self.test_y = [None]
        self.output_names = [None]

    def compute_scales(self):
        x_mean, x_std = 0.0, 0.0
        y_scales = [None] * self.O
        for i in range(self.O):
            x_mean += jnp.mean(self.train_x[i])
            x_std += jnp.std(self.train_x[i])
            y_scales[i] = (jnp.mean(self.train_y[i]), jnp.std(self.train_y[i]))

        self.x_scale = (x_mean / self.O, x_std / self.O)
        self.y_scales = y_scales

    def downscale(self, x, y):
        xo = [
            (xi - self.x_scale[0]) / self.x_scale[1] if xi is not None else xi
            for xi in x
        ]
        yo = [
            (yi - self.y_scales[i][0]) / self.y_scales[i][1] if yi is not None else yi
            for i, yi in enumerate(y)
        ]
        return xo, yo

    def upscale(self, x, y):
        xo = [
            (xi * self.x_scale[1] + self.x_scale[0]) if xi is not None else xi
            for xi in x
        ]
        yo = [
            (yi * self.y_scales[i][1] + self.y_scales[i][0]) if yi is not None else yi
            for i, yi in enumerate(y)
        ]
        return xo, yo

    def upscale_variance(self, var):

        varo = [
            vi * self.y_scales[i][1] ** 2 if vi is not None else vi
            for i, vi in enumerate(var)
        ]
        return varo


class WeatherDataSet(MODataSet):
    def __init__(self, data_dir: str):
        """
        Makes weather dataset.

        Args:
            data_dir (str): path to data dirctory.
        """
        super().__init__(data_dir)

    def load_data(self):
        data = loadmat(os.path.join(self.data_dir, "weatherdata.mat"))

        all_x = [jnp.array(x[0].flatten()) for x in data["xT"]]
        all_y = [jnp.array(y[0].flatten()) for y in data["yT"]]
        #%%

        train_x = copy.deepcopy(all_x)
        train_y = copy.deepcopy(all_y)

        train_x1 = []
        train_y1 = []
        test_x1 = []
        test_y1 = []
        for i, xi in enumerate(all_x[1]):
            if not (10.2 < xi and xi < 10.8):

                train_x1.append(xi)
                train_y1.append(all_y[1][i])
            else:
                test_x1.append(xi)
                test_y1.append(all_y[1][i])

        train_x2 = []
        train_y2 = []
        test_x2 = []
        test_y2 = []
        for i, xi in enumerate(all_x[2]):
            if not 13.5 < xi < 14.2:
                train_x2.append(xi)
                train_y2.append(all_y[2][i])
            else:
                test_x2.append(xi)
                test_y2.append(all_y[2][i])

        train_x[1] = jnp.array(train_x1)
        train_x[2] = jnp.array(train_x2)
        train_y[1] = jnp.array(train_y1)
        train_y[2] = jnp.array(train_y2)

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = [None, jnp.array(test_x1), jnp.array(test_x2), None]
        self.test_y = [None, jnp.array(test_y1), jnp.array(test_y2), None]
        self.output_names = ["Bramblemet", "Cambermet", "Chimet", "Sotonmet"]

