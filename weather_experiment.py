from scipy.io import loadmat
import matplotlib.pyplot as plt
from nvkm.utils import l2p, make_zg_grids, RMSE, NMSE, gaussian_NLPD
from nvkm.models import MOVarNVKM
from nvkm.experiments import WeatherDataSet

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import copy
import pickle
import os


def main(args):
    data = WeatherDataSet(args.data_dir)
    keys = jrnd.split(jrnd.PRNGKey(args.key), 6)
    plot_path = os.path.join("plots", args.f_name)
    model_path = os.path.join("pretrained_models", args.f_name)

    O = data.O
    C = len(args.Nvgs)

    zu = jnp.linspace(-args.zurange, args.zurange, args.Nvu).reshape(-1, 1)
    lsu = zu[1][0] - zu[0][0]

    tgs, lsgs = make_zg_grids(args.zgrange, args.Nvgs)

    model = MOVarNVKM(
        [tgs] * O,
        zu,
        (data.strain_x, data.strain_y),
        q_pars_init=None,
        q_initializer_pars=args.q_frac,
        q_init_key=keys[0],
        lsgs=[lsgs] * O,
        noise=[args.noise] * O,
        ampgs=[args.ampgs] * O,
        alpha=[[3 / (args.zgrange[i]) ** 2 for i in range(C)]] * O,
        lsu=lsu,
        ampu=1.0,
        N_basis=args.Nbasis,
    )

    model.fit(
        args.Nits, args.lr, args.Nbatch, args.Ns, dont_fit=["lsu", "noise"], key=keys[1]
    )
    model.fit(
        int(args.Nits / 10),
        args.lr,
        args.Nbatch,
        args.Ns,
        dont_fit=["q_pars", "ampgs", "lsgs", "ampu", "lsu"],
        key=keys[5],
    )
    model.save(model_path + "_model.pkl")

    axs = model.plot_samples(
        jnp.linspace(-args.zurange, args.zurange, 300),
        [jnp.linspace(-args.zurange, args.zurange, 300)] * O,
        args.Ns,
        return_axs=True,
        key=keys[2],
    )
    axs[2].scatter(data.stest_x[1], data.stest_y[1], c="red", alpha=0.3)
    axs[3].scatter(data.stest_x[2], data.stest_y[2], c="red", alpha=0.3)
    plt.savefig(plot_path + "_samples.pdf")
    plt.show()

    model.plot_filters(
        jnp.linspace(-max(args.zgrange), max(args.zgrange), 60),
        10,
        save=plot_path + "_filters.pdf",
        key=keys[3],
    )

    spreds = model.predict(data.stest_x, 50, key=keys[4])
    _, pred_mean = data.upscale(data.stest_x, spreds[0])
    pred_var = data.upscale_variance(spreds[1])

    fig, axs = plt.subplots(2, 1, figsize=(5, 5))
    for i in range(2):
        axs[i].plot(
            data.test_x[i + 1], data.test_y[i + 1], c="black", ls=":", label="Val. Data"
        )
        axs[i].plot(data.test_x[i + 1], pred_mean[i + 1], c="green", label="Pred. Mean")
        axs[i].fill_between(
            data.test_x[i + 1],
            pred_mean[i + 1] + 2 * jnp.sqrt(pred_var[i + 1]),
            pred_mean[i + 1] - 2 * jnp.sqrt(pred_var[i + 1]),
            alpha=0.1,
            color="green",
            label="$\pm 2 \sigma$",
        )

    plt.savefig(plot_path + "_preds.pdf")

    print(f"Cambermet NMSE: {NMSE(pred_mean[1], data.test_y[1]):.2f}")
    print(f"Chimet NMSE: {NMSE(pred_mean[2], data.test_y[2]):.2f}")
    print(
        f"Cambermet NLPD: {gaussian_NLPD(pred_mean[1], pred_var[1], data.test_y[1]):.2f}"
    )
    print(
        f"Chimet NLPD: {gaussian_NLPD(pred_mean[2], pred_var[2], data.test_y[2]):.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather MO experiment.")
    parser.add_argument(
        "--Nvu",
        default=100,
        type=int,
        help="Number of inducing points for input process.",
    )
    parser.add_argument(
        "--Nvgs",
        default=[15, 10, 6],
        nargs="+",
        type=int,
        help="List of number inducing points for each VK.",
    )
    parser.add_argument(
        "--zgrange",
        default=[0.463, 0.372, 0.239],
        nargs="+",
        type=float,
        help="List of widths for each VK.",
    )
    parser.add_argument(
        "--zurange",
        default=2.0,
        type=float,
        help="Range of inducing points for input process.",
    )
    parser.add_argument(
        "--Nits", default=10000, type=int, help="Number of training iterations."
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--Nbatch", default=80, type=int, help="Batch size.")
    parser.add_argument(
        "--Nbasis", default=30, type=int, help="Number of basis functions."
    )
    parser.add_argument(
        "--Ns", default=10, type=int, help="Number of samples for bound estimate."
    )
    parser.add_argument(
        "--ampgs",
        default=[5.0, 5.0, 5.0],
        nargs="+",
        type=float,
        help="Initial VK amplitudes.",
    )
    parser.add_argument(
        "--q_frac",
        default=0.8,
        type=float,
        help="Amount of initial variational covariance.",
    )
    parser.add_argument("--noise", default=0.05, type=float, help="Initial noise.")
    parser.add_argument(
        "--f_name", default="weather", type=str, help="Name for saving."
    )
    parser.add_argument("--data_dir", default="data", type=str, help="Data directory.")
    parser.add_argument("--key", default=102, type=int, help="Random seed.")
    args = parser.parse_args()
    main(args)
