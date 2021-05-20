from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit
import pandas as pd
import matplotlib.pyplot as plt
import scipy as osp
import copy
from functools import partial
import os
from scipy.io import loadmat
from .models import EQApproxGP


#%%


def generate_vdp_data(
    mus=[2.0, 1.0, 0.1, 0.0],
    reps=10,
    noise=0.1,
    path="/Users/magnus/Documents/phd/code/repos/nvkm/data/vdp",
    impute=True,
    N=500,
    N_te=50,
    key=jrnd.PRNGKey(1),
):
    keys = jrnd.split(key, 4)

    gp1D = EQApproxGP(z=None, v=None, amp=1.0, ls=5.0, noise=0.0001, N_basis=50, D=1)

    for mu in mus:
        mu_dir = path + "/mu" + str(mu).replace(".", "")
        try:
            os.mkdir(mu_dir)
        except FileExistsError:
            pass

        for i in range(reps):
            keys = jrnd.split(keys[0], 4)

            def gp_forcing(t):
                return gp1D.sample(jnp.array([t]), 1, key=keys[1]).flatten()

            def vdp(t, z):
                x, y = z
                return [y, mu * (1 - x ** 2) * y - x + gp_forcing(t)]

            t = jnp.linspace(0, 100, N)
            sol = osp.integrate.solve_ivp(vdp, [0, 100], [0, 0], t_eval=t)

            y = sol.y[0] + noise * jrnd.normal(key=keys[2], shape=(N,))
            x = (t - jnp.mean(t)) / jnp.std(t)
            y = (y - jnp.mean(y)) / jnp.std(y)
            if impute:
                tidx1 = jrnd.randint(keys[3], (1,), 0, N - 50)[0]
            else:
                tidx1 = N - 50

            tidx2 = tidx1 + N_te
            x_test, y_test = x[tidx1:tidx2], y[tidx1:tidx2]
            x_train, y_train = (
                jnp.hstack((x[:tidx1], x[tidx2:])),
                jnp.hstack((y[:tidx1], y[tidx2:])),
            )
            pd.DataFrame({"x_train": x_train, "y_train": y_train}).to_csv(
                mu_dir + "/rep" + str(i) + "train.csv"
            )
            pd.DataFrame({"x_test": x_test, "y_test": y_test}).to_csv(
                mu_dir + "/rep" + str(i) + "test.csv"
            )


def generate_duffing_data(
    path="/Users/magnus/Documents/phd/code/repos/nvkm/data/duffing",
):
    keys = jrnd.split(jrnd.PRNGKey(0), 5)
    for i in range(10):
        gp1D = EQApproxGP(
            z=None, v=None, amp=1.0, ls=2.0, noise=0.0001, N_basis=50, D=1
        )

        @jit
        def gp_forcing(t):
            return gp1D.sample(t, 1, key=keys[0]).flatten()

        def duffing(t, z, a=1, b=-1, c=1.0):
            x, y = z
            # y = dx/dt
            return [y, -a * y - b * x - c * x ** 3 + gp_forcing(t)]

        N = 500

        t = jnp.linspace(0, 100, N)
        sol = osp.integrate.solve_ivp(duffing, [0, 100], [0.0, 0.0], t_eval=t)

        y = sol.y[0]
        x = (t - jnp.mean(t)) / jnp.std(t)
        y = (y - jnp.mean(y)) / jnp.std(y) + 0.1 * jrnd.normal(keys[3], (N,))

        N_te = 40
        N_te_rand = 300
        txs = jrnd.randint(keys[1], (1,), minval=0, maxval=N - N_te)

        all_idx = jnp.arange(N)
        t_idx_imp = jnp.arange(txs, N_te + txs)
        t_idx_rand = jnp.sort(
            jrnd.choice(keys[2], all_idx[~jnp.isin(all_idx, t_idx_imp)], (N_te_rand,))
        )
        t_idx = jnp.hstack((t_idx_imp, t_idx_rand))

        x_test, y_test = x[t_idx], y[t_idx]
        train_idx = ~jnp.isin(all_idx, t_idx)
        x_train, y_train = x[train_idx], y[train_idx]

        pd.DataFrame({"x_train": x_train, "y_train": y_train}).to_csv(
            path + "/rep" + str(i) + "train.csv"
        )
        pd.DataFrame({"x_test": x_test, "y_test": y_test}).to_csv(
            path + "/rep" + str(i) + "test.csv"
        )
        keys = jrnd.split(keys[4], 5)


def generate_mo_duffing_data(
    path="/Users/magnus/Documents/phd/code/repos/nvkm/data/mo_duffing",
):
    keys = jrnd.split(jrnd.PRNGKey(1), 4)
    for k in range(5):
        gp1D = EQApproxGP(
            z=None, v=None, amp=1.0, ls=2.0, noise=0.0001, N_basis=50, D=1
        )

        @jit
        def gp_forcing(t):
            return gp1D.sample(t, 1, key=keys[0]).flatten()

        def duffing(t, z, a=1, b=-1, c=0.1, d=1.0):
            x, y = z
            # y = dx/dt
            return [y, -a * y - b * x - c * x ** 3 + d * gp_forcing(t)]

        N = 500

        t = jnp.linspace(0, 150, N)
        ys = []
        for b, d in [(1, 1.0), (0.6, 1.0), (0.8, 3.0)]:

            sol = osp.integrate.solve_ivp(
                partial(duffing, b=-b, d=d), [0, 150], [0.0, 0.0], t_eval=t
            )
            y = sol.y[0]
            y = (y - jnp.mean(y)) / jnp.std(y) + 0.1 * jrnd.normal(keys[1], (N,))
            ys.append(y)
            keys = jrnd.split(keys[0], 4)

        x = (t - jnp.mean(t)) / jnp.std(t)

        N_te = 50

        all_idx = jnp.arange(N)
        x_test, y_test, x_train, y_train = [0.0] * 3, [0.0] * 3, [0.0] * 3, [0.0] * 3
        all_idx = jnp.arange(N)
        run_idx = jnp.arange(N - N_te)
        for i in range(3):
            txs = jrnd.choice(keys[i], run_idx, (1,))
            t_idx = jnp.arange(txs, N_te + txs)
            run_idx = run_idx[~jnp.isin(run_idx, jnp.arange(txs - N_te, N_te + txs))]

            x_test[i], y_test[i] = x[t_idx], ys[i][t_idx]
            train_idx = ~jnp.isin(all_idx, t_idx)
            x_train[i], y_train[i] = x[train_idx], ys[i][train_idx]

        train_dict = {f"x{o}_train": x_train[o] for o in range(3)}
        train_dict.update({f"y{o}_train": y_train[o] for o in range(3)})

        test_dict = {f"x{o}_test": x_test[o] for o in range(3)}
        test_dict.update({f"y{o}_test": y_test[o] for o in range(3)})

        pd.DataFrame(train_dict).to_csv(path + "/rep" + str(k) + "train.csv")
        pd.DataFrame(test_dict).to_csv(path + "/rep" + str(k) + "test.csv")
        print(k, "done!")
        keys = jrnd.split(keys[4], 5)


def load_vdp_data(mu, rep, data_dir="data"):
    path = data_dir + "/vdp" + "/mu" + str(mu).replace(".", "") + "/rep" + str(rep)
    tr_df = pd.read_csv(path + "train.csv")
    te_df = pd.read_csv(path + "test.csv")
    return (
        jnp.array(tr_df["x_train"]),
        jnp.array(tr_df["y_train"]),
        jnp.array(te_df["x_test"]),
        jnp.array(te_df["y_test"]),
    )


def load_duffing_data(rep, data_dir="data"):
    path = data_dir + "/duffing/rep" + str(rep)
    tr_df = pd.read_csv(path + "train.csv")
    te_df = pd.read_csv(path + "test.csv")
    return (
        jnp.array(tr_df["x_train"]),
        jnp.array(tr_df["y_train"]),
        jnp.array(te_df["x_test"]),
        jnp.array(te_df["y_test"]),
    )


def load_mo_duffing_data(rep, data_dir="data"):
    path = data_dir + "/mo_duffing/rep" + str(rep)
    tr_df = pd.read_csv(path + "train.csv")
    te_df = pd.read_csv(path + "test.csv")
    x_train = [jnp.array(tr_df[f"x{i}_train"]) for i in range(1, 3)]
    y_train = [jnp.array(tr_df[f"y{i}_train"]) for i in range(1, 3)]
    x_test = [jnp.array(te_df[f"x{i}_test"]) for i in range(1, 3)]
    y_test = [jnp.array(te_df[f"y{i}_test"]) for i in range(1, 3)]
    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )


class MODataSet:
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
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def load_data(self):
        data = loadmat(self.data_dir + "/weatherdata.mat")

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


# %%
class ExchangeDataSet(MODataSet):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def load_data(self):
        train_df = pd.read_csv(self.data_dir + "/fx/fx_train.csv", index_col=0)
        test_df = pd.read_csv(self.data_dir + "/fx/fx_test.csv", index_col=0)

        xs = []
        ys = []
        o_names = []
        x = jnp.array(train_df["year"])
        for key in train_df.keys():
            if key != "year":
                o_names.append(key)
                yi = jnp.array(train_df[key])
                xs.append(x[~jnp.isnan(yi)])
                ysi = yi[~jnp.isnan(yi)]
                ys.append(jnp.array(ysi))

        self.train_x, self.train_y, self.output_names = xs, ys, o_names

        xte = [None] * len(xs)
        yte = [None] * len(xs)
        for o in test_df.keys():
            if o != "year":
                yi = jnp.array(test_df[o])
                xte[o_names.index(o)] = jnp.array(test_df["year"][~jnp.isnan(yi)])
                yte[o_names.index(o)] = yi[~jnp.isnan(yi)]

        self.test_x, self.test_y = xte, yte

    # # def vdp(t, z):
    # #     x, y = z
    # #     return [y, -4 * y - x + gp_forcing(t) ** p]

    # # def vdp(t, z):
    # #     x, y = z
    # #     return [y, mu * (1 - x ** 2) * y - x + gp_forcing(t) ** 3]

    # # def vdp(t, z):
    # #     x, y = z
    # #     # y = dx/dt
    # #     return [y, -y - x + 10.0*jnp.cos(x/2) ** 2 + 0.1*gp_forcing(t)]

    # @jit
    # def G1(x, a=1.0, b=1.0, alpha=1.0):
    #     return jnp.exp(-alpha * x ** 2) * (jnp.sin(2 * x))

    # @jit
    # def G2(x, a=1.0, b=1.0, alpha=1.0):
    #     return jnp.exp(-alpha * x ** 2) * (-jnp.sin(2.0 * x) ** 2 + x / 2)

    # @jit
    # def G3(x, a=1.0, b=1.0, alpha=1.0):
    #     return jnp.exp(-alpha * x ** 2) * (-jnp.cos(3.0 * x))

    # @partial(jit, static_argnums=(1, 2, 3))
    # def trapz_int(t, h, x, N, dim=1, decay=4):
    #     tau = jnp.linspace(t - decay, t + decay, N)
    #     ht = h(t - tau)
    #     xt = x(tau)
    #     return jnp.trapz(ht * xt, x=tau, axis=0)

    # N = 500
    # xu = jnp.linspace(-20, 20, N)
    # Nint = 100

    # fyc1 = jit(lambda x: trapz_int(x, G1, gp_forcing, Nint, decay=2.0))
    # yc1 = vmap(fyc1)(xu)
    # fyc2a = jit(lambda x: trapz_int(x, G2, gp_forcing, Nint, decay=2.0))
    # fyc2b = jit(lambda x: trapz_int(x, G1, gp_forcing, Nint, decay=2.0))

    # fyc3 = jit(lambda x: trapz_int(x, G3, gp_forcing, Nint, decay=2.0))

    # yc2 = vmap(fyc2a)(xu) * vmap(fyc2b)(xu)
    # yc3 = vmap(fyc2a)(xu) * vmap(fyc2b)(xu) * vmap(fyc3)(xu)
    # y = yc1 + yc2

    # u = jnp.array([gp_forcing(ti) for ti in t])
    # u = (u - jnp.mean(u)) / jnp.std(u)
