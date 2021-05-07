#%%
from nvkm.models import MOVarNVKM, EQApproxGP
from nvkm.utils import l2p, NMSE, make_zg_grids, gaussian_NLPD

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
import pandas as pd
import argparse
from functools import partial
import scipy as osp
from sklearn.model_selection import train_test_split
import pickle
import GPy
import argparse

parser = argparse.ArgumentParser(description="EEG MO experiment.")
parser.add_argument("--Nvu", default=70, type=int)
parser.add_argument("--Nvgs", default=[15], nargs="+", type=int)
parser.add_argument("--zgrange", default=[0.3], nargs="+", type=float)
parser.add_argument("--zurange", default=2.0, type=float)
parser.add_argument("--Nits", default=1000, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--Nbatch", default=30, type=int)
parser.add_argument("--Nbasis", default=30, type=int)
parser.add_argument("--Ns", default=5, type=int)
parser.add_argument("--ampgs", default=[2.0], nargs="+", type=float)
parser.add_argument("--q_frac", default=0.7, type=float)
parser.add_argument("--noise", default=0.1, type=float)
parser.add_argument("--f_name", default="vdp", type=str)
parser.add_argument("--mode", default="expr", type=str)
parser.add_argument("--key", default=1, type=int)
parser.add_argument("--mus", default=[10.0, 1.0, 0.1, 0.01], nargs="+", type=float)
args = parser.parse_args()

Nbatch = args.Nbatch
Nbasis = args.Nbasis
noise = args.noise
Nits = args.Nits
Nvu = args.Nvu
Nvgs = args.Nvgs
zgran = args.zgrange
zuran = args.zurange
Ns = args.Ns
lr = args.lr
q_frac = args.q_frac
f_name = args.f_name
ampgs = args.ampgs
key = args.key
mus = args.mus
mode = args.mode
print(args)

# Nbatch = 50
# Nbasis = 30
# noise = 0.1
# Nits = 500
# Nvu = 70
# Ns = 5
# lr = 1e-2
# q_frac = 0.8
# f_name = "vdp"
# mode = "expr"
# Nvgs = [15, 10, 8]
# zgran = [0.3, 0.2, 0.2]
# ampgs = [2.0, 2.0, 2.0]
# zuran = 2.0
# key = 1
# mus = [10.0, 1.0, 0.1, 0.01]

keys = jrnd.split(jrnd.PRNGKey(key), 5)

#%%
gp1D = EQApproxGP(z=None, v=None, amp=1.0, ls=5.0, noise=0.0001, N_basis=50, D=1)
#%%
def gp_forcing(t, key=jrnd.PRNGKey(1)):
    return gp1D.sample(jnp.array([t]), 1, key=key).flatten()


def vdp(t, z, mu=1.0, key=jrnd.PRNGKey(1)):
    x, y = z
    return [y, mu * (1 - x ** 2) * y - x + gp_forcing(t, key=key)]


def generate_vdp_data(mu, noise=0.1, impute=True, N=500, N_te=50, key=jrnd.PRNGKey(1)):
    keys = jrnd.split(key, 3)
    x = jnp.linspace(0, 200, N)
    sol = osp.integrate.solve_ivp(
        partial(vdp, mu=mu, key=keys[0]), [0, 200], [0, 0], t_eval=x
    )
    y = sol.y[0] + noise * jrnd.normal(key=keys[1], shape=(N,))
    x = (x - jnp.mean(x)) / jnp.std(x)
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
    return x_train, y_train, x_test, y_test


# %%

if mode == "demo":
    x_train, y_train, x_test, y_test = generate_vdp_data(0.01, impute=True)

    ode_kernel = GPy.kern.EQ_ODE2(input_dim=2)
    gpy_model = GPy.models.GPRegression(
        jnp.vstack((x_train, jnp.ones_like(x_train))).T,
        y_train.reshape(-1, 1),
        ode_kernel,
    )
    gpy_model.optimize_restarts(num_restarts=10)
    yp_train = gpy_model.predict(
        jnp.vstack((jnp.linspace(-2, 2, 500), jnp.ones(500))).T
    )

    #%%
    fig = plt.figure(figsize=(10, 2))
    plt.plot(jnp.linspace(-2, 2, 500), yp_train[0])
    plt.plot(jnp.linspace(-2, 2, 500), yp_train[1])
    plt.scatter(x_train, y_train)
    plt.scatter(x_test, y_test)
    plt.show()
    # %%
    O = 1
    C = len(Nvgs)

    zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
    lsu = zu[1][0] - zu[0][0]

    tgs, lsgs = make_zg_grids(zgran, Nvgs)

    model = MOVarNVKM(
        [tgs] * O,
        zu,
        ([x_train], [y_train]),
        q_pars_init=None,
        q_initializer_pars=q_frac,
        q_init_key=keys[0],
        lsgs=[lsgs] * O,
        ampgs=[ampgs] * O,
        noise=[noise] * O,
        alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]] * O,
        lsu=lsu,
        ampu=1.0,
        N_basis=Nbasis,
    )
    # %%
    model.fit(
        Nits, lr, Nbatch, Ns, dont_fit=["lsgs", "ampu", "lsu", "noise"], key=keys[1],
    )
    print(model.noise)
    print(model.ampu)
    print(model.lsu)
    print(model.ampgs)
    print(model.lsgs)
    # %%
    axs = model.plot_samples(
        jnp.linspace(-zuran, zuran, 300),
        [jnp.linspace(-zuran, zuran, 300)] * O,
        Ns,
        return_axs=True,
        key=keys[2],
    )
    axs[1].scatter(x_test, y_test, c="red", s=2.0)
    plt.savefig(f_name + "fit_samps.pdf")
    plt.show()

    model.plot_filters(
        jnp.linspace(-max(zgran), max(zgran), 60),
        10,
        save=f_name + "fit_filters.pdf",
        key=keys[3],
    )
    # %%
    preds = model.sample([x_test], 30, key=keys[4])[0]
    #%%
    lfm_mean, lfm_std = gpy_model.predict(jnp.vstack((x_test, jnp.ones_like(x_test))).T)
    lfm_std = jnp.sqrt(lfm_std).flatten()
    lfm_mean = lfm_mean.flatten()
    # %%
    pred_mean = jnp.mean(preds, axis=1)
    pred_std = jnp.std(preds, axis=1)
    plt.plot(x_test, pred_mean, c="green", label="Pred. Mean")
    plt.fill_between(
        x_test,
        pred_mean + 2 * pred_std,
        pred_mean - 2 * pred_std,
        alpha=0.1,
        color="green",
        label="$\pm 2 \sigma$",
    )
    plt.plot(x_test, lfm_mean, c="blue", label="Pred. Mean")
    plt.fill_between(
        x_test,
        lfm_mean + 2 * lfm_std,
        lfm_mean - 2 * lfm_std,
        alpha=0.1,
        color="blue",
        label="$\pm 2 \sigma$",
    )
    plt.plot(x_test, y_test, c="black", ls=":", label="Val. Data")
    plt.savefig(f_name + "main.pdf")
    plt.show()
# %%
else:
    NMSEs = {
        "NVKM": [],
        "ODE1": [],
    }
    NLPDs = {
        "NVKM": [],
        "ODE1": [],
    }
    for mu in mus:
        keys = jrnd.split(keys[0], 5)
        x_train, y_train, x_test, y_test = generate_vdp_data(
            mu, impute=True, key=keys[1]
        )

        ode_kernel = GPy.kern.EQ_ODE1(input_dim=2)
        gpy_model = GPy.models.GPRegression(
            jnp.vstack((x_train, jnp.ones_like(x_train))).T,
            y_train.reshape(-1, 1),
            ode_kernel,
        )
        gpy_model.optimize_restarts(num_restarts=10)
        ode_mean, ode_var = gpy_model.predict(
            jnp.vstack((x_test, jnp.ones_like(x_test))).T
        )
        nmse = NMSE(ode_mean, y_test)
        nlpd = gaussian_NLPD(ode_mean, ode_var, y_test)

        NMSEs["ODE1"].append(nmse)
        NLPDs["ODE1"].append(nlpd)

        O = 1
        C = len(Nvgs)
        zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
        lsu = zu[1][0] - zu[0][0]

        tgs, lsgs = make_zg_grids(zgran, Nvgs)

        model = MOVarNVKM(
            [tgs] * O,
            zu,
            ([x_train], [y_train]),
            q_pars_init=None,
            q_initializer_pars=q_frac,
            q_init_key=keys[2],
            lsgs=[lsgs] * O,
            ampgs=[ampgs] * O,
            noise=[noise] * O,
            alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]] * O,
            lsu=lsu,
            ampu=1.0,
            N_basis=Nbasis,
        )
        model.fit(
            Nits,
            lr,
            Nbatch,
            Ns,
            dont_fit=["lsgs", "ampu", "lsu", "noise"],
            key=keys[3],
        )
        preds = model.sample([x_test], 50, key=keys[4])[0]
        pred_mean = jnp.mean(preds, axis=1)
        pred_var = jnp.var(preds, axis=1) + model.noise[0] ** 2

        nmse = NMSE(pred_mean, y_test)
        nlpd = gaussian_NLPD(pred_mean, pred_var, y_test)

        NMSEs["NVKM"].append(nmse)
        NLPDs["NVKM"].append(nlpd)

    res = {"NLPD": NLPDs, "NMSE": NMSEs}
    print(res)
    with open(f_name + "res.pkl", "wb") as f:
        pickle.dump(res, f)

    # %%

