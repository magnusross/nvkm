#%%
from nvkm.utils import l2p, make_zg_grids, RMSE, gaussian_NLPD
from nvkm.models import IOMOVarNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pickle
import os


def main(args):
    keys = jrnd.split(jrnd.PRNGKey(args.key), 7)
    plot_path = os.path.join("plots", args.f_name)
    model_path = os.path.join("pretrained_models", args.f_name)

    data = pd.read_csv(os.path.join(args.data_dir, "water_tanks.csv"))
    y_mean, y_std = data["yEst"].mean(), data["yEst"].std()
    u_mean, u_std = data["uEst"].mean(), data["uEst"].std()
    t_mean, t_std = data["Ts"].mean(), data["Ts"].std()

    tt = jnp.array((data["Ts"] - t_mean) / t_std)
    utrain = (tt, jnp.array((data["uEst"] - u_mean) / u_std))
    ytrain = ([tt], [jnp.array((data["yEst"] - y_mean) / y_std)])

    t_offset = 20.0
    utest = (
        tt + t_offset * jnp.ones(len(data)),
        jnp.array((data["uVal"] - u_mean) / u_std),
    )
    ytest = (
        [tt + t_offset * jnp.ones(len(data))],
        [jnp.array((data["yVal"] - y_mean) / y_std)],
    )

    udata = (jnp.hstack((utrain[0], utest[0])), jnp.hstack((utrain[1], utest[1])))

    C = len(args.Nvgs)

    zu = jnp.hstack(
        (
            jnp.linspace(-args.zurange, args.zurange, 150),
            t_offset + jnp.linspace(-args.zurange, args.zurange, 150),
        )
    ).reshape(-1, 1)

    lsu = zu[0][0] - zu[1][0]

    tgs, lsgs = make_zg_grids(args.zgrange, args.Nvgs)
    # %%

    model = IOMOVarNVKM(
        [tgs],
        zu,
        udata,
        ytrain,
        q_pars_init=None,
        q_initializer_pars=args.q_frac,
        q_init_key=keys[0],
        lsgs=[lsgs],
        ampgs=[args.ampgs],
        alpha=[[3 / (args.zgrange[i]) ** 2 for i in range(C)]],
        lsu=lsu,
        ampu=1.0,
        N_basis=args.Nbasis,
        u_noise=args.noise,
        noise=[args.noise],
    )

    model.fit(
        args.Nits,
        args.lr,
        args.Nbatch,
        args.Ns,
        dont_fit=["lsu", "noise", "u_noise"],
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
    # %%
    tp_train = jnp.linspace(-args.zurange, args.zurange, 400)
    tp_test = tp_train + t_offset
    axs = model.plot_samples(tp_train, [tp_train], 10, return_axs=True, key=keys[2])
    axs[0].set_xlim([-args.zurange, args.zurange])
    axs[1].set_xlim([-args.zurange, args.zurange])
    plt.savefig(plot_path + "_samps_train.pdf")
    #%%
    axs = model.plot_samples(tp_test, [tp_test], 10, return_axs=True, key=keys[3])
    axs[0].set_xlim([t_offset - args.zurange, t_offset + args.zurange])
    axs[1].set_xlim([t_offset - args.zurange, t_offset + args.zurange])
    axs[1].plot(ytest[0][0], ytest[1][0], c="black", ls=":")
    plt.savefig(plot_path + "_samps_test.pdf")
    # axs[1].xrange(18, 22)
    #%%

    p_samps = model.sample(ytest[0], 50, key=keys[4])
    u_samps_tr, p_samps_tr = model.joint_sample(utrain[0], ytrain[0], 50, key=keys[5])

    #%%
    scaled_samps = p_samps[0] * y_std + y_mean
    pred_mean = jnp.mean(scaled_samps, axis=1)
    pred_std = jnp.std(scaled_samps, axis=1)
    pred_var = pred_std ** 2 + y_std ** 2 * model.noise[0] ** 2

    scaled_samps_tr = p_samps_tr[0] * y_std + y_mean
    pred_mean_tr = jnp.mean(scaled_samps_tr, axis=1)
    pred_std_tr = jnp.std(scaled_samps_tr, axis=1)
    pred_var_tr = pred_std_tr ** 2 + y_std ** 2 * model.noise[0] ** 2

    u_scaled_samps_tr = u_samps_tr * u_std + u_mean
    u_pred_mean_tr = jnp.mean(u_scaled_samps_tr, axis=1)
    u_pred_std_tr = jnp.std(u_scaled_samps_tr, axis=1)
    u_pred_var_tr = u_pred_std_tr ** 2 + u_std ** 2 * model.u_noise ** 2

    rmse = RMSE(pred_mean, jnp.array(data["yVal"]))
    nlpd = gaussian_NLPD(pred_mean, pred_var, jnp.array(data["yVal"]))

    rmse_tr = (
        RMSE(pred_mean_tr, jnp.array(data["yEst"]))
        + RMSE(u_pred_mean_tr, jnp.array(data["uEst"]))
    ) / 2
    nlpd_tr = (
        gaussian_NLPD(pred_mean_tr, pred_var_tr, jnp.array(data["yEst"]))
        + gaussian_NLPD(u_pred_mean_tr, u_pred_var_tr, jnp.array(data["uEst"]))
    ) / 2
    #%%
    print("Train RMSE: %.3f" % rmse_tr)
    print("Train NLPD: %.3f" % nlpd_tr)

    print("RMSE: %.3f" % rmse)
    print("NLPD: %.3f" % nlpd)
    #%%
    fig = plt.figure(figsize=(12, 4))
    plt.plot(data["Ts"], data["yVal"], c="black", ls=":", label="Val. Data")
    plt.plot(data["Ts"], pred_mean, c="green", label="Pred. Mean")
    plt.fill_between(
        data["Ts"],
        pred_mean + 2 * pred_std,
        pred_mean - 2 * pred_std,
        alpha=0.1,
        color="green",
        label="$\pm 2 \sigma$",
    )
    plt.text(0, 12, "$e_{RMS}$ = %.2f" % rmse)
    plt.text(0, 13, "$e_{NLPD}$ = %.2f" % nlpd)
    plt.xlabel("time (s)")
    plt.ylabel("output (V)")
    plt.legend()
    plt.savefig(plot_path + "_preds.pdf")
    plt.show()
    #%%
    tf = jnp.linspace(-max(args.zgrange), max(args.zgrange), 100)
    model.plot_filters(tf, 15, save=plot_path + "_filters.pdf", key=keys[6])


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water tank IO experiment.")
    parser.add_argument(
        "--Nvu",
        default=150,
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
        default=[0.375, 0.303, 0.128],
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
    parser.add_argument("--f_name", default="tanks", type=str, help="Name for saving.")
    parser.add_argument("--key", default=103, type=int, help="Random seed.")
    parser.add_argument("--data_dir", default="data", type=str, help="Data directory.")
    args = parser.parse_args()
    main(args)
