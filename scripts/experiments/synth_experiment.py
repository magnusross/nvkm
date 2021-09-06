from nvkm.models import MOVarNVKM, EQApproxGP
from nvkm.utils import l2p, NMSE, make_zg_grids, gaussian_NLPD
from nvkm.experiments import load_volterra_data

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap
import pandas as pd
import argparse
from functools import partial
import scipy as osp

import pickle
from pathlib import Path
import os


def main(args):
    keys = jrnd.split(jrnd.PRNGKey(args.rep), 5)

    plot_path = os.path.join("plots", args.f_name)
    model_path = os.path.join("pretrained_models", args.f_name)
    preds_path = os.path.join(args.preds_dir, "synth")

    x_train, y_train, x_test, y_test = load_volterra_data(
        args.rep, data_dir=os.path.join(args.data_dir, "volt")
    )

    keys = jrnd.split(keys[0], 5)
    C = len(args.Nvgs)
    zu = jnp.linspace(-args.zurange, args.zurange, args.Nvu).reshape(-1, 1)
    lsu = zu[1][0] - zu[0][0]

    tgs, lsgs = make_zg_grids(args.zgrange, args.Nvgs)

    model = MOVarNVKM(
        [tgs],
        zu,
        ([x_train], [y_train]),
        q_frac=args.q_frac,
        key=keys[0],
        lsgs=[lsgs],
        ampgs=[args.ampgs],
        noise=[args.noise],
        alpha=[[3 / (args.zgrange[i]) ** 2 for i in range(C)]],
        lsu=lsu,
        ampu=1.0,
        N_basis=args.Nbasis,
    )

    model.fit(
        args.Nits,
        args.lr,
        args.Nbatch,
        args.Ns,
        dont_fit=["lsu", "noise"],
        key=keys[1],
    )

    model.fit(
        int(args.Nits / 10),
        args.lr,
        args.Nbatch,
        args.Ns,
        dont_fit=["q_pars", "ampgs", "lsgs", "ampu", "lsu"],
        key=keys[2],
    )

    model.save(model_path + "_model.pkl")
    _ = model.plot_samples(
        jnp.linspace(-args.zurange, args.zurange, 300),
        [jnp.linspace(-args.zurange, args.zurange, 300)],
        10,
        return_axs=True,
        key=keys[3],
    )
    plt.savefig(plot_path + "_samples.pdf")
    plt.show()

    t = jnp.linspace(-args.zurange, args.zurange, 300)
    preds = model.predict([t], 30)
    mean, var = preds[0][0], preds[1][0]

    _ = plt.figure(figsize=(10, 2))
    plt.scatter(x_train, y_train, c="black", s=10, alpha=0.5)
    plt.plot(t, mean, c="green")
    plt.scatter(x_test, y_test, c="red", s=10, alpha=0.5)
    plt.fill_between(
        t,
        mean - 2 * jnp.sqrt(var),
        mean + 2 * jnp.sqrt(var),
        alpha=0.1,
        color="green",
    )
    plt.savefig(plot_path + "_preds.pdf")
    plt.show()

    model.plot_filters(
        jnp.linspace(-max(args.zgrange), max(args.zgrange), 100),
        10,
        key=keys[3],
    )
    plt.savefig(plot_path + "_filters.pdf")

    m, s = model.predict([x_test], 50)

    test_nmse = NMSE(m[0], y_test)
    test_nlpd = gaussian_NLPD(m[0], s[0], y_test)
    print("Test NMSE:", test_nmse)
    print("Test NLPD:", test_nlpd)

    pred_name = "rep" + str(args.rep) + "predictions.csv"
    odir = Path(os.path.join(preds_path, "nvkmC" + str(C)))
    odir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"x_test": x_test, "y_test": y_test, "pred_mean": m[0], "pred_var": s[0]}
    ).to_csv(os.path.join(odir, pred_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sythetic data experiment.")
    parser.add_argument(
        "--Nvu",
        default=88,
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
        default=[1.0, 0.8, 0.8],
        nargs="+",
        type=float,
        help="List of widths for each VK.",
    )
    parser.add_argument(
        "--zurange",
        default=22.0,
        type=float,
        help="Range of inducing points for input process.",
    )
    parser.add_argument(
        "--Nits", default=10000, type=int, help="Number of training iterations."
    )
    parser.add_argument("--lr", default=2e-3, type=float, help="Learning rate.")
    parser.add_argument("--Nbatch", default=80, type=int, help="Batch size.")
    parser.add_argument(
        "--Nbasis", default=30, type=int, help="Number of basis functions."
    )
    parser.add_argument(
        "--Ns", default=10, type=int, help="Number of samples for bound estimate."
    )
    parser.add_argument(
        "--ampgs",
        default=[0.5, 0.5, 0.5],
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
    parser.add_argument("--noise", default=0.03, type=float, help="Initial noise.")
    parser.add_argument("--f_name", default="synth", type=str, help="Name for saving.")
    parser.add_argument("--rep", default=0, type=int, help="Repeat number.")
    parser.add_argument("--data_dir", default="data", type=str, help="Data directory.")
    parser.add_argument(
        "--preds_dir", default="preds", type=str, help="Predictions directory."
    )
    args = parser.parse_args()
    main(args)
