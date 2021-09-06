
from nvkm.utils import l2p, make_zg_grids, make_zg_grids1D
from nvkm.models import MOVarNVKM, SepHomogMOVarNVKM, SepMOVarNVKM

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

# to change default colormap
plt.rcParams["image.cmap"] = "Set2"
# to change default color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Set2.colors)
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif"],
    }
)
if __name__ == "__main__":
    keys = jrnd.split(jrnd.PRNGKey(0), 5)
    data_dir = "data"
    keys = jrnd.split(keys[0], 5)
    zurange = 22.0
    Nvgs = [10] * 3
    zgran = [1.5] * 3
    Nvu = 88
    C = len(Nvgs)
    zu = jnp.linspace(-zurange, zurange, Nvu).reshape(-1, 1)
    lsu = zu[1][0] - zu[0][0]

    zgs, lsgs = make_zg_grids(zgran, Nvgs)
    zgs1, lsgs1 = make_zg_grids1D(zgran, Nvgs)

    model = MOVarNVKM(
        [zgs],
        zu,
        ([None], [None]),
        q_frac=0.8,
        lsgs=[lsgs1],
        ampgs=[[1.0] * C],
        key=keys[0],
        noise=[0.05],
        alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]],
        lsu=lsu,
        ampu=1.0,
        N_basis=20,
    )

    hom_model = SepHomogMOVarNVKM(
        [zgs1],
        zu,
        ([None], [None]),
        q_frac=0.8,
        lsgs=[lsgs1],
        ampgs=[[1.0] * C],
        key=keys[0],
        noise=[0.05],
        alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]],
        lsu=lsu,
        ampu=1.0,
        N_basis=20,
    )

    sep_model = SepMOVarNVKM(
        [zgs1],
        zu,
        ([None], [None]),
        q_frac=0.8,
        lsgs=[lsgs1],
        ampgs=[[1.0] * C],
        key=keys[0],
        noise=[0.05],
        alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]],
        lsu=lsu,
        ampu=1.0,
        N_basis=20,
    )
    #%%
    ts = [jnp.linspace(-22, 22, 300)]
    fig, axs = plt.subplots(3, 1, figsize=(7, 8))
    for i, m in enumerate([model, sep_model, hom_model]):
        s = m.sample(ts, 3, key=jrnd.PRNGKey(1))
        axs[i].plot(ts[0], s[0])
        axs[i].set_xlabel("$t$")
        axs[i].set_ylabel("$f(t)$")

    axs[0].set_title("Full.")
    axs[1].set_title("Separable")
    axs[2].set_title("Homog.")
    plt.tight_layout()
    plt.savefig("samples.pdf")
    plt.show()

