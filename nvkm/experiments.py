from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import pandas as pd
import matplotlib.pyplot as plt
import scipy as osp
from functools import partial
import os
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
