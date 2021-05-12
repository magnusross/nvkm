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


Nbatch = 50
Nbasis = 30
noise = 0.1
Nits = 500
Nvu = 70
Ns = 5
lr = 1e-2
q_frac = 0.8
f_name = "vdp"
mode = "demo"
Nvgs = [15, 10, 8]
zgran = [0.3, 0.2, 0.2]
ampgs = [2.0, 2.0, 2.0]
zuran = 2.0
key = 1
mus = [10.0, 1.0, 0.1, 0.01]
# #%%
# x_train, y_train, x_test, y_test = load_vdp_data(0.0, 6)
# plt.scatter(x_train, y_train)
# plt.scatter(x_test, y_test)
# plt.show()
#     x_train, y_train, x_test, y_test = generate_vdp_data(1.0, impute=True)

#     ode_kernel = GPy.kern.EQ_ODE2(input_dim=2)
#     gpy_model = GPy.models.GPRegression(
#         jnp.vstack((x_train, jnp.ones_like(x_train))).T,
#         y_train.reshape(-1, 1),
#         ode_kernel,
#     )
#     gpy_model.optimize_restarts(num_restarts=10)
#     yp_train = gpy_model.predict(
#         jnp.vstack((jnp.linspace(-2, 2, 500), jnp.ones(500))).T
#     )

#     #%%
#     fig = plt.figure(figsize=(10, 2))
#     plt.plot(jnp.linspace(-2, 2, 500), yp_train[0])
#     plt.plot(jnp.linspace(-2, 2, 500), yp_train[1])
#     plt.scatter(x_train, y_train)
#     plt.scatter(x_test, y_test)
#     plt.show()
#     # %%
#     O = 1
#     C = len(Nvgs)

#     zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
#     lsu = zu[1][0] - zu[0][0]

#     tgs, lsgs = make_zg_grids(zgran, Nvgs)

#     model = MOVarNVKM(
#         [tgs] * O,
#         zu,
#         ([x_train], [y_train]),
#         q_pars_init=None,
#         q_initializer_pars=q_frac,
#         q_init_key=keys[0],
#         lsgs=[lsgs] * O,
#         ampgs=[ampgs] * O,
#         noise=[noise] * O,
#         alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]] * O,
#         lsu=lsu,
#         ampu=1.0,
#         N_basis=Nbasis,
#     )
#     # %%
#     model.fit(
#         Nits, lr, Nbatch, Ns, dont_fit=["lsgs", "ampu", "lsu", "noise"], key=keys[1],
#     )
#     print(model.noise)
#     print(model.ampu)
#     print(model.lsu)
#     print(model.ampgs)
#     print(model.lsgs)
#     # %%
#     axs = model.plot_samples(
#         jnp.linspace(-zuran, zuran, 300),
#         [jnp.linspace(-zuran, zuran, 300)] * O,
#         Ns,
#         return_axs=True,
#         key=keys[2],
#     )
#     axs[1].scatter(x_test, y_test, c="red", s=2.0)
#     plt.savefig(f_name + "fit_samps.pdf")
#     plt.show()

#     model.plot_filters(
#         jnp.linspace(-max(zgran), max(zgran), 60),
#         10,
#         save=f_name + "fit_filters.pdf",
#         key=keys[3],
#     )
#     # %%
#     preds = model.sample([x_test], 30, key=keys[4])[0]
#     #%%
#     lfm_mean, lfm_std = gpy_model.predict(jnp.vstack((x_test, jnp.ones_like(x_test))).T)
#     lfm_std = jnp.sqrt(lfm_std).flatten()
#     lfm_mean = lfm_mean.flatten()
#     # %%
#     pred_mean = jnp.mean(preds, axis=1)
#     pred_std = jnp.std(preds, axis=1)
#     plt.plot(x_test, pred_mean, c="green", label="Pred. Mean")
#     plt.fill_between(
#         x_test,
#         pred_mean + 2 * pred_std,
#         pred_mean - 2 * pred_std,
#         alpha=0.1,
#         color="green",
#         label="$\pm 2 \sigma$",
#     )
#     plt.plot(x_test, lfm_mean, c="blue", label="Pred. Mean")
#     plt.fill_between(
#         x_test,
#         lfm_mean + 2 * lfm_std,
#         lfm_mean - 2 * lfm_std,
#         alpha=0.1,
#         color="blue",
#         label="$\pm 2 \sigma$",
#     )
#     plt.plot(x_test, y_test, c="black", ls=":", label="Val. Data")
#     plt.savefig(f_name + "main.pdf")
#     plt.show()# %%
