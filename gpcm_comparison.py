import lab.torch as B
from gpcm.experiment import run
import torch

import numpy as np
import argparse

from gpcm.experiment import setup, run, build_models, train_models, analyse_models
from gpcm.model import train_vi
from wbml.experiment import WorkingDirectory
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--train-method",
#     choices=["vi", "laplace"],
#     default="vi",
#     nargs="?",
# )
# parser.add_argument(
#     "--model",
#     choices=["gpcm", "gprv", "cgpcm"],
#     default=["gpcm"],
#     nargs="+",
# )
# parser.add_argument("path", nargs="*")
# args = parser.parse_args()

# wd = WorkingDirectory(
#     "_gpcm_experiments",
#     "mag",
#     *args.path,
#     seed=1,
# )


noise = 0.05
t_plot = B.linspace(torch.float64, -20, 20, 200)

# Setup true model and GPCM models.

window = 1  # like 2x my zgrange?
n_u = 30
scale = 0.1
n_z = 80
print(scale)
# Sample data.
preds_dir = "preds/volt"


def load_duffing_data(rep, data_dir="data"):
    path = data_dir + "/volt/"
    tr_df = pd.read_csv(path + "rep" + str(rep) + "train.csv")
    te_df = pd.read_csv(path + "rep" + str(rep) + "test.csv")
    return (
        torch.Tensor(tr_df["x_train"]),
        torch.Tensor(tr_df["y_train"]),
        torch.Tensor(te_df["x_test"]),
        torch.Tensor(te_df["y_test"]),
    )


args, wd = setup("duffing")


from nvkm.utils import NMSE, gaussian_NLPD
import jax.numpy as jnp


for i in range(0, 10):

    x_train, y_train, x_test, y_test = load_duffing_data(i)
    models = build_models(
        args.model,
        noise=noise,
        window=window,
        scale=scale,
        t=x_train,
        y=y_train,
        n_u=n_u,
        n_z=n_z,
    )

    dists = train_models(train_vi, models, wd=wd, iters=200, fix_noise=args.fix_noise)

    model = models[0][2](models[0][1])

    mean, std = model.predict(dists[0], x_test)
    var = jnp.array(std.numpy()) ** 2 + jnp.array(model.noise)
    with open("gpcm_out.txt", "a") as f:
        print(i, window, scale, file=f)
        print(NMSE(jnp.array(mean.numpy()), jnp.array(y_test.numpy())), file=f)
        print(
            gaussian_NLPD(jnp.array(mean.numpy()), var, jnp.array(y_test.numpy()),),
            file=f,
        )
    f_name = "rep" + str(i) + "predictions.csv"
    odir = Path(preds_dir + "/gpcm")
    odir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "x_test": x_test,
            "y_test": y_test,
            "pred_mean": jnp.array(mean.numpy()),
            "pred_var": var,
        }
    ).to_csv(odir / f_name)
    print("saved results at: ", odir / f_name)
# analyse_models(
#     models,
#     dists,
#     t=x_train,
#     y=y_train,
#     wd=wd,
#     t_plot=t_plot,
#     truth=(x_test, y_test),
# )
# run(
#     wd=wd,
#     args=args,
#     noise=noise,
#     window=window,
#     scale=scale,
#     t=x_train,
#     y=y_train,
#     n_u=n_u,
#     n_z=n_z,
#     t_plot=t_plot,
#     truth=(x_test, y_test),
#     y_range={"kernel": (-0.5, 1.5)},
# )
