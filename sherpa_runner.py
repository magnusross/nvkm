import sherpa
import sherpa

parameters = [
    sherpa.Choice(name="param_a", range=[1, 2, 3]),
    sherpa.Continuous(name="param_b", range=[0, 1]),
]

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=10)
scheduler = sherpa.schedulers.LocalScheduler()
results = sherpa.optimize(
    parameters=parameters,
    algorithm=algorithm,
    lower_is_better=True,
    filename="/Users/magnus/Documents/phd/code/repos/nvkm/sherpa_trial.py",
    scheduler=scheduler,
    max_concurrent=2,
    verbose=1,
)
# parameters = [
#     sherpa.Continuous(name="noise", range=[0.01, 0.2]),
#     sherpa.Continuous(name="zgran", range=[0.1, 1]),
# ]

# algorithm = sherpa.algorithms.RandomSearch(max_num_trials=10)
# scheduler = sherpa.schedulers.LocalScheduler()
# results = sherpa.optimize(
#     parameters=parameters,
#     algorithm=algorithm,
#     lower_is_better=True,
#     scheduler=scheduler,
#     filename="/Users/magnus/Documents/phd/code/repos/nvkm/sherpa_trial.py",
#     output_dir=".",
#     max_concurrent=1,
#     verbose=1,
# )

