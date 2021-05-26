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

alt_m_dir = "preds/volt/"
data_dir = "data/volt/"
model_names = [n for n in os.listdir(alt_m_dir) if not n.startswith(".")]
mus = os.listdir(data_dir)
# nlpd_df = pd.DataFrame(
#     columns=(onp.repeat(onp.array(mus), 2), onp.array(["m", "s"] * len(mus))),
#     dtype=object,
# )
# nmse_df = pd.DataFrame(
#     columns=(onp.repeat(onp.array(mus), 2), onp.array(["m", "s"] * len(mus))),
#     dtype=object,
# )
nmse_df = pd.DataFrame(columns=["score"], dtype=object)
df = pd.DataFrame(index=model_names, columns=["NLPD", "NMSE"])
std_df = pd.DataFrame(columns=["NLPD", "NMSE"])
reps = 10
for model_name in model_names:
    smus = os.listdir(data_dir)
    # fig, axs = plt.subplots(len(smus), reps, figsize=(50, 20))
    # nmse_df.loc[model_name] = jnp.ones((4, 2))
    # nlpd_df.loc[model_name] = jnp.ones((4, 2))

    if "ode1" in model_name:
        continue

    rep_nmses = []
    rep_nlpds = []
    for i in range(reps):
        path_pred = alt_m_dir + model_name + "/rep" + str(i) + "predictions.csv"
        path_true = data_dir + "/rep" + str(i) + "test.csv"
        if "nvkm" in model_name:
            try:
                pred_df = pd.read_csv(path_pred, dtype=float)
                print(pred_df)
                x = jnp.array(pred_df["x_test"])
                y = jnp.array(pred_df["pred_mean"])
                var = jnp.array(pred_df["pred_var"])
                y_test = jnp.array(pred_df["y_test"])
                rmse1 = NMSE(y, y_test)
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

            if jnp.any(var < 0.0):
                print("Negative varince for: " + path_pred)
                continue

        true_df = pd.read_csv(path_true)
        true_y = jnp.array(true_df["y_test"])
        nmse = NMSE(y, true_y)
        nlpd = gaussian_NLPD(y, var, true_y)
        if "nvkm" in model_name:
            print(i, nmse, nlpd)
            assert jnp.all(jnp.isclose(y_test, true_y))

        rep_nmses.append(nmse)
        rep_nlpds.append(gaussian_NLPD(y, var, jnp.array(true_df["y_test"])))

        if i == 0:
            fig = plt.figure(figsize=(10, 2))
            plt.plot(x, y)
            plt.scatter(x, true_y, s=1)
            plt.title(model_name)
            plt.show()
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
    print(model_name + " done!")
    print(rep_nmses)
    try:
        df.loc[model_name]["NMSE"] = jnp.median(jnp.array(rep_nmses))
        df.loc[model_name]["NLPD"] = jnp.median(jnp.array(rep_nlpds))
    except:
        pass

    # plt.savefig("plots/mega_alt_results/" + model_name + ".pdf")
    # plt.close(fig)

print(df.sort_index())

# %%
