#%%
from nvkm.models import MOVarNVKM, EQApproxGP
from nvkm.utils import l2p, RMSE, NMSE, make_zg_grids, gaussian_NLPD
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
import pandas as pd
import argparse
from functools import partial
import scipy as osp
import pickle
import argparse
import os
import numpy as onp

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc("legend", fontsize=10)  # using a size in points
plt.rc("legend", fontsize="small")
#%%
def main():
    alt_m_dir = os.path.join("preds", "paper", "synth")
    data_dir = os.path.join("data", "volt")
    model_names = [n for n in os.listdir(alt_m_dir) if not n.startswith(".")]
    model_names.sort()

    df = pd.DataFrame(
        index=model_names, columns=["NMSE", "NMSE_std", "NLPD", "NLPD_std"]
    )

    reps = 10
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 6)
    axs1 = fig.add_subplot(gs[0, :4])
    axs2 = fig.add_subplot(gs[0, 4:])
    plot_i = 0
    lw = 0.7
    colors = ["red", "green", "blue", "orange", "purple"]
    names = dict(
        {"gpcm": "GPCM"}, **{f"nvkmC{i}": f"NVKM ($C={i}$)" for i in range(1, 5)}
    )
    for j, model_name in enumerate(model_names):
        rep_nmses = []
        rep_nlpds = []
        for i in range(reps):
            path_pred = os.path.join(
                alt_m_dir, model_name, "rep" + str(i) + "predictions.csv"
            )

            try:
                pred_df = pd.read_csv(path_pred, dtype=float)

            except FileNotFoundError:
                print("No file for: " + path_pred)
                continue
            x = jnp.array(pred_df["x_test"])
            y = jnp.array(pred_df["pred_mean"])
            var = jnp.array(pred_df["pred_var"])
            y_test = jnp.array(pred_df["y_test"])
            nmse = NMSE(y, y_test)
            nlpd = gaussian_NLPD(y, var, y_test)
            rep_nmses.append(nmse)
            rep_nlpds.append(nlpd)

            if i == plot_i:
                axs1.plot(x, y, label=names[model_name], c=colors[j], lw=lw, alpha=0.75)
                axs2.plot(x, y, c=colors[j], lw=lw)
                axs2.fill_between(
                    x,
                    y - 2 * jnp.sqrt(var),
                    y + 2 * jnp.sqrt(var),
                    alpha=0.2,
                    color=colors[j],
                )

        print(model_name + " done!")

        df.loc[model_name]["NMSE"] = jnp.mean(jnp.array(rep_nmses))
        df.loc[model_name]["NLPD"] = jnp.mean(jnp.array(rep_nlpds))
        df.loc[model_name]["NMSE_std"] = jnp.std(jnp.array(rep_nmses))
        df.loc[model_name]["NLPD_std"] = jnp.std(jnp.array(rep_nlpds))

    train_path = os.path.join(data_dir, "rep") + str(plot_i) + "train.csv"
    train_df = pd.read_csv(train_path)
    x_train, y_train = train_df["x_train"], train_df["y_train"]

    test_path = os.path.join(data_dir, "rep") + str(plot_i) + "test.csv"
    test_df = pd.read_csv(test_path)
    x_test, y_test = test_df["x_test"], test_df["y_test"]

    axs1.scatter(x_test, y_test, c="black", s=3, lw=0.5, alpha=0.5, label="Test")
    axs1.scatter(
        x_train, y_train, c="black", marker="x", s=3, lw=0.5, alpha=0.5, label="Train"
    )
    axs2.scatter(x_test, y_test, c="black", s=10, lw=1, alpha=0.5)
    axs2.scatter(x_train, y_train, c="black", marker="x", lw=1, s=10, alpha=0.5)
    axs1.set_xlabel("$t$")
    axs1.set_ylabel("$y(t)$")
    axs1.legend()
    axs2.set_xlim(-10, -7)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "paper", "synth.pdf"))
    plt.show()
    with pd.option_context("precision", 3):
        print(df.astype(float))


if __name__ == "__main__":
    main()
# %%
