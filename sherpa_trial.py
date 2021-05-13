import sherpa

client = sherpa.Client()
trial = client.get_trial()
# Model training
num_iterations = 10
for i in range(num_iterations):
    pseudo_objective = (
        trial.parameters["param_a"] / float(i + 1) * trial.parameters["param_b"]
    )
    client.send_metrics(trial=trial, iteration=i + 1, objective=pseudo_objective)


# import sherpa
# from nvkm.models import MOVarNVKM
# from nvkm.utils import l2p, NMSE, make_zg_grids, gaussian_NLPD
# from nvkm.experiments import ExchangeDataSet, WeatherDataSet

# import matplotlib.pyplot as plt
# import jax.numpy as jnp
# import jax.random as jrnd
# import pandas as pd
# import argparse


# client = sherpa.Client()
# trial = client.get_trial()

# Nbatch = 50
# Nbasis = 30
# noise = trial.parameters["noise"]
# Nits = 100
# Nvu = 100
# Ns = 5
# lr = 1e-2
# q_frac = 0.6
# f_name = "fx"
# data_dir = "data"
# Nvgs = [20]
# zgran = [trial.parameters["zgran"]]
# ampgs = [5.0]
# zuran = 2.0
# key = 1

# keys = jrnd.split(jrnd.PRNGKey(key), 5)

# data = WeatherDataSet(data_dir)

# O = len(data.output_names)
# C = len(Nvgs)

# zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
# lsu = zu[1][0] - zu[0][0]

# tgs, lsgs = make_zg_grids(zgran, Nvgs)


# model = MOVarNVKM(
#     [tgs] * O,
#     zu,
#     (data.strain_x, data.strain_y),
#     q_pars_init=None,
#     q_initializer_pars=q_frac,
#     q_init_key=keys[0],
#     lsgs=[lsgs] * O,
#     ampgs=[ampgs] * O,
#     noise=[noise] * O,
#     alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]] * O,
#     lsu=lsu,
#     ampu=1.0,
#     N_basis=Nbasis,
# )

# model.fit(
#     Nits, lr, Nbatch, Ns, dont_fit=["lsgs", "ampu", "lsu", "noise"], key=keys[1],
# )


# # print(model.noise)
# # print(model.ampu)
# # print(model.lsu)
# # print(model.ampgs)
# # print(model.lsgs)

# train_preds = model.predict(data.strain_x, 10)
# _, pred_mean = data.upscale(data.strain_x, train_preds[0])
# _, pred_var = data.upscale(data.strain_x, train_preds[1])

# train_total_nlpd = (
#     sum([gaussian_NLPD(pred_mean[i], pred_var[i], data.train_y[i]) for i in range(O)])
#     / O
# )
# train_total_nmse = sum([NMSE(pred_mean[i], data.train_y[i]) for i in range(O)]) / O

# client.send_metrics(trial=trial, iteration=1, objective=train_total_nlpd)
