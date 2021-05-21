#%%
from nvkm.models import MOVarNVKM, EQApproxGP
from nvkm.utils import l2p, RMSE, NMSE, make_zg_grids, gaussian_NLPD
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
import os
import numpy as onp

#%%

alt_m_dir = "preds/vdp/"
data_dir = "data/vdp/"
model_names = os.listdir(alt_m_dir)
mus = os.listdir(data_dir)
# nlpd_df = pd.DataFrame(
#     columns=(onp.repeat(onp.array(mus), 2), onp.array(["m", "s"] * len(mus))),
#     dtype=object,
# )
# nmse_df = pd.DataFrame(
#     columns=(onp.repeat(onp.array(mus), 2), onp.array(["m", "s"] * len(mus))),
#     dtype=object,
# )
nmse_df = pd.DataFrame(columns=os.listdir(data_dir), dtype=object)
nlpd_df = pd.DataFrame(columns=os.listdir(data_dir), dtype=object)
reps = 10
for model_name in model_names:
    smus = os.listdir(data_dir)
    # fig, axs = plt.subplots(len(smus), reps, figsize=(50, 20))
    # nmse_df.loc[model_name] = jnp.ones((4, 2))
    # nlpd_df.loc[model_name] = jnp.ones((4, 2))

    if "ode1" in model_name:
        continue

    for j, smu in enumerate(smus):
        rep_nmses = []
        rep_nlpds = []
        for i in range(reps):
            path_pred = (
                alt_m_dir + model_name + "/" + smu + "/rep" + str(i) + "predictions.csv"
            )
            path_true = data_dir + smu + "/rep" + str(i) + "test.csv"
            if "nvkm" in model_name:
                try:
                    pred_df = pd.read_csv(path_pred, dtype=float)
                    x = jnp.array(pred_df["x_test"])
                    y = jnp.array(pred_df["pred_mean"])
                    var = jnp.array(pred_df["pred_var"])
                except FileNotFoundError:
                    print("No file for: " + path_pred)
                    continue

            else:
                try:
                    pred_df = pd.read_csv(path_pred, header=None, dtype=float)
                    x = jnp.real(jnp.array(pred_df[1]))
                    y = jnp.real(jnp.array(pred_df[2]))
                    var = jnp.real(jnp.array(pred_df[3]))

                except FileNotFoundError:
                    print("No file for: " + path_pred)
                    continue
                except ValueError:
                    print("Imaginairy values for: " + path_pred)
                    continue

                if onp.any(var < 0.0):
                    print("Negative varince for: " + path_pred)
                    continue

            true_df = pd.read_csv(path_true)
            nmse = RMSE(y, jnp.array(true_df["y_test"]))
            nlpd = gaussian_NLPD(y, var, jnp.array(true_df["y_test"]))
            if "nvkm" in model_name:
                print(i, smu, nmse, nlpd)
            rep_nmses.append(nmse)
            rep_nlpds.append(gaussian_NLPD(y, var, jnp.array(true_df["y_test"])))

            # axs[j, i].plot(x, jnp.real(y))
            # axs[j, i].fill_between(
            #     x, y - 2 * jnp.sqrt(var), y + 2 * jnp.sqrt(var), alpha=0.3,
            # )
            # axs[j, i].plot(true_df["x_test"], true_df["y_test"])

            # axs[j, i].set_title(model_name + "mu" + smu + "r" + str(i))

        # nlpd_df.loc[model_name, (smu, "m")] = jnp.mean(jnp.array(rep_nlpds))
        # nlpd_df.loc[model_name, (smu, "s")] = jnp.std(jnp.array(rep_nlpds))

        # nmse_df.loc[model_name, (smu, "m")] = jnp.mean(jnp.array(rep_nmses))
        # nmse_df.loc[model_name, (smu, "s")] = jnp.std(jnp.array(rep_nmses))
        rep_nmses = jnp.sort(jnp.array(rep_nmses))
        rep_nlpds = jnp.sort(jnp.array(rep_nlpds))
        nmse_df.loc[model_name, smu] = (
            "{:.2f}".format(jnp.mean(jnp.array(rep_nmses))),
            "{:.2f}".format(jnp.std(jnp.array(rep_nmses)) / jnp.sqrt(len(rep_nmses))),
        )
        nlpd_df.loc[model_name, smu] = (
            "{:.2f}".format(jnp.mean(jnp.array(rep_nlpds))),
            "{:.2f}".format(jnp.std(jnp.array(rep_nlpds)) / jnp.sqrt(len(rep_nlpds))),
        )

    print(model_name + " done!")
    # plt.savefig("plots/mega_alt_results/" + model_name + ".pdf")
    # plt.close(fig)

print(nmse_df.sort_index())
print(nlpd_df.sort_index())
# %%

# %%
