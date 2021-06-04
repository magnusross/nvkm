# Learning Nonparametric Volterra Kernels with Gaussian Processes

Python + Jax implementation of the nonparametric Volterra kernels model (NVKM).

Requirements
------------
You can install the requirements by running,

```
pip install -r requirements.txt
```

Note that only python version 3.8 has been tested. The code runs significantly faster on the GPU if one is available. To use the GPU, first follow the instructions [here](https://github.com/google/jax#installation) to get the GPU version of Jax, and then install the rest of the requirements. 

Generate paper plots
--------------------

You can generate the plots from the real data experiments in the paper, by running

```
python make_paper_plots.py
```

which loads the pre-trained models shown in the paper, makes predictions then generates the plots. 

You can generate the synthetic data plots and table by running,
```
python make_synth_results.py
```
which loads predictions and calculates the relevant statistics and makes the plots.

Train paper models
------------------

You can train the models with the settings shown in the paper by running,

```
python synth_experiment.py
python water_tank_experiment.py
python weather_experiment.py
```

which will produce a variety of plots (in `plots` directory) and metrics, as well as a `.pkl` file (in the `pretrained_models` directory) containing the model. *Warning* this takes quite a while to run especially if not on the GPU. 

Training other models
---------------------
You can train models with your own settings using command line options, for example
```
python water_tank_experiment.py --Nits 1000 --Nvgs 15 --ampgs 5.0 --zgrange 0.35
```
would train a model on the tanks experiment for 1000 iterations using one Volterra kernel with 15 inducing points and width 0.35. 

You can see the list of available options by running, for example
```
python water_tank_experiment.py -h
```
