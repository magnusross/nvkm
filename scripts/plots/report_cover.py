#%%
from nvkm.models import SepMOVarNVKM
from nvkm.utils import l2p, make_zg_grids1D, gaussian_NLPD


import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrnd
from matplotlib import cm

#%%
if __name__ == "__main__":
    keys = jrnd.split(jrnd.PRNGKey(1), 5)
    O = 5
    zurange = 22.0
    Nvgs = [15] * 3
    zgran = [1.5] * 3
    Nvu = 88
    C = len(Nvgs)
    zu = jnp.linspace(-zurange, zurange, Nvu).reshape(-1, 1)
    lsu = zu[1][0] - zu[0][0]
    zgs, lsgs = make_zg_grids1D(zgran, Nvgs)

    model = SepMOVarNVKM(
        [zgs] * O,
        zu,
        ([None], [None]),
        q_frac=1.0,
        lsgs=[lsgs] * O,
        ampgs=[[1.0] * 3] * O,
        key=keys[0],
        noise=[0.05] * O,
        alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]] * O,
        lsu=lsu,
        ampu=1.0,
        N_basis=20,
    )

    ts = jnp.linspace(-10, 10, 200)
    s = model.sample([ts] * O, 1, keys[1])

    fig, axs = plt.subplots(5, 1, figsize=(10, 8), frameon=False)
    for i in range(O):
        axs[i].plot(ts, s[i][:, 0], c=cm.get_cmap("Set2")(i))
        axs[i].axis("off")
    plt.tight_layout()
    fig.patch.set_visible(False)
    plt.savefig("cover.pdf", transparent=True)
    plt.show()

# %%
