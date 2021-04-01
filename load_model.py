from nvkm.utils import generate_EQ_data
from nvkm.models import NVKM, VariationalNVKM
from nvkm.vi import IndependentGaussians, gaussain_likelihood
from jax.config import config
import jax.numpy as jnp
import jax.random as jrnd
import pickle 

config.update("jax_enable_x64", True)

# with open('logs/model1.pkl') as f:
#     model = pickle.load(f)
model = jnp.load('logs/model1.pkl.npz', allow_pickle=True)

def load_var_nvkm(f_name, var_dist=IndependentGaussians, likelihood=gaussain_likelihood):
    raw_dict = jnp.load(f_name, allow_pickle=True)
    d = {key:raw_dict[key].item() for key in raw_dict}
    model = VariationalNVKM(d['zgs'], d['zu'], d['data'], var_dist, likelihood=likelihood)
    for key in d.keys():
        if key not in ["zgs", "zu", "data"]:
            setattr(model, key, d[key])
    return model

model = load_var_nvkm('logs/model1.pkl.npz')
model.plot_filters(jnp.linspace(-6, 6, 100), 10, save="plots/pickled.png")