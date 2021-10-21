from nvkm.models import IOMOVarNVKM, load_io_model, load_mo_model
from nvkm.utils import (
    l2p,
    make_zg_grids,
    RMSE,
    gaussian_NLPD,
)
from nvkm.experiments import WeatherDataSet
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pickle
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "cm"


def plot_tanks():
    data = pd.read_csv(os.path.join("data", "water_tanks.csv"))
    y_mean, y_std = data["yEst"].mean(), data["yEst"].std()
    u_mean, u_std = data["uEst"].mean(), data["uEst"].std()
    t_mean, t_std = data["Ts"].mean(), data["Ts"].std()

    model = load_io_model(
        os.path.join("pretrained_models", "paper", "tank_paper_model.pkl")
    )

    tf = jnp.linspace(-0.5, 0.5, 100)

    tfs = [
        [jnp.vstack((tf for j in range(gp.D))).T for gp in model.g_gps[i]]
        for i in range(model.O)
    ]
    raw_g_samps = model.sample_diag_g_gps(tfs, 50, jrnd.split(jrnd.PRNGKey(2), 2))[0]
    g_samps = [
        y_std * model.ampgs[0][i] * jnp.exp(-model.alpha[0][i] * (tf) ** 2) * gi.T
        for i, gi in enumerate(raw_g_samps)
    ]

    g_means = [jnp.mean(gi, axis=0) for gi in g_samps]
    g_std = [jnp.std(gi, axis=0) for gi in g_samps]

    fig = plt.figure(constrained_layout=True, figsize=(12, 4))
    gs = fig.add_gridspec(5, model.C[0])

    titles = ["$G_{1, 1} (t)$", "$G_{1, 2} (t, t)$", "$G_{1, 3} (t, t, t)$"]
    for i in range(model.C[0]):
        axs = fig.add_subplot(gs[:2, i])
        axs.plot(tf * t_std, g_means[i], c="red", label="$\mu$")
        axs.fill_between(
            tf * t_std,
            g_means[i] - 2 * g_std[i],
            g_means[i] + 2 * g_std[i],
            color="red",
            alpha=0.3,
            label="$\pm 2 \sigma$",
        )
        axs.set_ylabel(titles[i], labelpad=0)
        axs.set_xlabel("t", labelpad=0)
        # axs.set_ylim(-12, 22)
        if i == 0:
            axs.legend()

    axs = fig.add_subplot(gs[2:, :])
    to = model.data[0][0][1024:][1::4]
    preds = model.predict([to], 10)
    s_mean = y_std * preds[0][0] + y_mean
    s_var = y_std ** 2 * preds[1][0]
    axs.plot(data["Ts"][1::4], s_mean, c="green", label="$\mu$")
    axs.fill_between(
        data["Ts"][1::4],
        s_mean - 2 * jnp.sqrt(s_var),
        s_mean + 2 * jnp.sqrt(s_var),
        color="green",
        alpha=0.3,
        label="$\pm 2 \sigma$",
    )
    axs.plot(data["Ts"], data["yVal"], c="black", ls="--", label="Test")
    axs.set_xlabel("Time (s)", loc="left", labelpad=0)
    axs.set_ylabel("Output (V)")
    axs.legend()
    plt.savefig(os.path.join("plots", "paper", "tanks.pdf"))


def plot_weather():
    data_set = WeatherDataSet("data")
    model = load_mo_model(
        os.path.join("pretrained_models", "paper", "weather_paper_model.pkl")
    )

    train_x = [d[1::4] for d in data_set.train_x]
    strain_x = [d[1::4] for d in data_set.strain_x]
    train_spreds = model.predict(strain_x, 15)
    _, train_pred_mean = data_set.upscale(strain_x, train_spreds[0])
    train_pred_var = data_set.upscale_variance(train_spreds[1])

    spreds = model.predict(data_set.stest_x, 15)
    _, pred_mean = data_set.upscale(data_set.stest_x, spreds[0])
    pred_var = data_set.upscale_variance(spreds[1])

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(1, 3):
        x = jnp.hstack((data_set.test_x[i], train_x[i]))
        sort_idx = jnp.argsort(x)
        x = x[sort_idx]
        y = jnp.hstack((pred_mean[i], train_pred_mean[i]))
        v = jnp.hstack((pred_var[i], train_pred_var[i]))
        y = y[sort_idx]
        v = v[sort_idx]
        axs[i - 1].plot(x, y, c="green", label="$\mu$")
        axs[i - 1].fill_between(
            x,
            y + 2 * jnp.sqrt(v),
            y - 2 * jnp.sqrt(v),
            alpha=0.2,
            color="green",
            label="$\pm 2 \sigma$",
        )
        axs[i - 1].scatter(
            data_set.test_x[i],
            data_set.test_y[i],
            c="deepskyblue",
            alpha=0.3,
            marker="o",
            label="Test",
            s=15,
        )
        axs[i - 1].scatter(
            data_set.train_x[i],
            data_set.train_y[i],
            c="black",
            alpha=0.3,
            marker="+",
            label="Train",
            s=15,
        )
        axs[i - 1].set_xlabel("Time (days)")
        axs[i - 1].set_ylabel(f"{data_set.output_names[i]} Temperature ($\degree$C)")

    plt.setp(axs, ylim=axs[0].get_ylim())
    axs[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "paper", "weather.pdf"))


if __name__ == "__main__":
    print("Plotting weather...")
    plot_weather()
    print("Plotting tanks...")
    plot_tanks()
