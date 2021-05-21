#%%
from nvkm.models import MOVarNVKM, EQApproxGP
from nvkm.utils import l2p, NMSE, make_zg_grids, gaussian_NLPD
from nvkm.experiments import load_vdp_data

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
import pandas as pd
import argparse
from functools import partial
import scipy as osp
import pickle
import GPy
import argparse
from pathlib import Path

# parser = argparse.ArgumentParser(description="EEG MO experiment.")
# parser.add_argument("--Nvu", default=70, type=int)
# parser.add_argument("--Nvgs", default=[15], nargs="+", type=int)
# parser.add_argument("--zgrange", default=[0.3], nargs="+", type=float)
# parser.add_argument("--zurange", default=2.0, type=float)
# parser.add_argument("--Nits", default=1000, type=int)
# parser.add_argument("--lr", default=1e-2, type=float)
# parser.add_argument("--Nbatch", default=30, type=int)
# parser.add_argument("--Nbasis", default=30, type=int)
# parser.add_argument("--Ns", default=5, type=int)
# parser.add_argument("--ampgs", default=[2.0], nargs="+", type=float)
# parser.add_argument("--q_frac", default=0.7, type=float)
# parser.add_argument("--noise", default=0.1, type=float)
# parser.add_argument("--f_name", default="vdp", type=str)
# parser.add_argument("--mode", default="expr", type=str)
# parser.add_argument("--rep", default=0, type=int)
# parser.add_argument("--mus", default=[2.0, 1.0, 0.1, 0.0], nargs="+", type=float)
# parser.add_argument("--data_dir", default="data", type=str)
# parser.add_argument("--preds_dir", default="preds", type=str)
# args = parser.parse_args()

# Nbatch = args.Nbatch
# Nbasis = args.Nbasis
# noise = args.noise
# Nits = args.Nits
# Nvu = args.Nvu
# Nvgs = args.Nvgs
# zgran = args.zgrange
# zuran = args.zurange
# Ns = args.Ns
# lr = args.lr
# q_frac = args.q_frac
# f_name = args.f_name
# ampgs = args.ampgs
# rep = args.rep
# mus = args.mus
# mode = args.mode
# data_dir = args.data_dir
# preds_dir = args.preds_dir
# print(args)

Nbatch = 50
Nbasis = 30
noise = 0.1
Nits = 5000
Nvu = 70
Ns = 5
lr = 3e-3
q_frac = 0.8
f_name = "vdp"
mode = "expr"
Nvgs = [15]
zgran = [0.3]
ampgs = [2.0]
zuran = 2.0
rep = 1
mus = [2.0]
data_dir = "data"
preds_dir = "preds"

keys = jrnd.split(jrnd.PRNGKey(rep), 5)

# %%
# %%

NMSEs = {
    "NVKM": [],
    "EQ": [],
}
NLPDs = {
    "NVKM": [],
    "EQ": [],
}
for mu in mus:
    keys = jrnd.split(keys[0], 5)
    x_train, y_train, x_test, y_test = load_vdp_data(mu, rep, data_dir=data_dir)

    eq_kernel = GPy.kern.ExpQuad(input_dim=1)
    gpy_model = GPy.models.GPRegression(
        x_train.reshape(-1, 1), y_train.reshape(-1, 1), eq_kernel,
    )
    gpy_model.optimize_restarts(num_restarts=10)
    eq_mean, eq_var = gpy_model.predict(x_train.reshape(-1, 1))
    nmse = NMSE(eq_mean, y_test)
    nlpd = gaussian_NLPD(eq_mean, eq_var, y_test)

    NMSEs["EQ"].append(nmse)
    NLPDs["EQ"].append(nlpd)

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
        Nits, lr, Nbatch, Ns, dont_fit=["lsu", "noise"], key=keys[2],
    )

    axs = model.plot_samples(
        jnp.linspace(-zuran, zuran, 300),
        [jnp.linspace(-zuran, zuran, 300)] * O,
        Ns,
        return_axs=True,
        key=keys[2],
    )
    plt.show()
    model.fit(
        Nits,
        lr,
        Nbatch,
        Ns,
        dont_fit=["q_pars", "ampgs", "lsgs", "ampu", "lsu"],
        key=keys[3],
    )

    model.save("plots/" + f_name + str(mu) + ".pkl")
    print(model.noise)
    print(model.ampu)
    print(model.lsu)
    print(model.ampgs)
    print(model.lsgs)

    axs = model.plot_samples(
        jnp.linspace(-zuran, zuran, 300),
        [jnp.linspace(-zuran, zuran, 300)] * O,
        Ns,
        return_axs=True,
        key=keys[2],
    )
    axs[1].scatter(x_test, y_test, c="red", s=2.0)
    plt.savefig("plots/" + f_name + str(mu) + ".pdf")
    plt.show()
    quit()
    preds = model.sample([x_test], 50, key=keys[4])[0]
    pred_mean = jnp.mean(preds, axis=1)
    pred_var = jnp.var(preds, axis=1) + model.noise[0] ** 2

    f_name = "rep" + str(rep) + "predictions.csv"
    odir = Path(preds_dir + "/nvkm/nvkmC" + str(C) + "/mu" + str(mu).replace(".", ""))
    odir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"x_test": x_test, "pred_mean": pred_mean, "pred_var": pred_var}
    ).to_csv(odir / f_name)

    nmse = NMSE(pred_mean, y_test)
    nlpd = gaussian_NLPD(pred_mean, pred_var, y_test)

    NMSEs["NVKM"].append(nmse)
    NLPDs["NVKM"].append(nlpd)

res = {"NLPD": NLPDs, "NMSE": NMSEs}
print(res)
with open(f_name + "res.pkl", "wb") as f:
    pickle.dump(res, f)

    # %%

