#%%
from nvkm.models import MOVarNVKM, EQApproxGP
from nvkm.utils import l2p, NMSE, make_zg_grids, gaussian_NLPD, map2matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import jax.numpy as jnp
import jax.random as jrnd
import copy

#%%

Nvgs = [15, 8]
Nvu = 40
O = 3
C = len(Nvgs)
zuran = 2.0
zgran = [0.4, 0.4]
ampgs = [[1.0, 5.0], [2.0, 4.0], [2.0, 4.2]]
keys = jrnd.split(jrnd.PRNGKey(4), 6)
zu = jnp.linspace(-zuran, zuran, Nvu).reshape(-1, 1)
lsu = zu[1][0] - zu[0][0]

tgs, lsgs = make_zg_grids(zgran, Nvgs)

model = MOVarNVKM(
    [tgs] * O,
    zu,
    None,
    q_pars_init=None,
    q_initializer_pars=0.0001,
    q_init_key=keys[2],
    lsgs=[lsgs] * O,
    ampgs=ampgs,
    noise=[0.01] * O,
    alpha=[[3 / (zgran[i]) ** 2 for i in range(C)]] * O,
    lsu=lsu,
    ampu=1.0,
    N_basis=50,
)
#%%
modelc1 = copy.deepcopy(model)
modelc1.C = [1, 1, 1]

#%%
fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(3, 7)


Nim = 100
tv1 = jnp.linspace(-zgran[1] - 0.3, zgran[1] + 0.3, Nim)
tm2 = jnp.meshgrid(tv1, tv1)
tv2 = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T


vs = model.q_of_v.sample(model.q_pars, 1, keys[0])

g_samps = model.sample_diag_g_gps(
    [[tv1.reshape(-1, 1), tv2]] * O, 1, jrnd.split(jrnd.PRNGKey(1), 2)
)
g_posl = [4, 11]
ys = [
    (
        jnp.exp(-1 * model.alpha[i][1] * jnp.sum((tv2) ** 2, axis=1))
        * g_samps[i][1].flatten()
    )
    for i in [0, 1, 2]
]
maxval = max(jnp.max(ys[0]), jnp.max(ys[1]))
minval = min(jnp.min(ys[0]), jnp.min(ys[1]))
for i in range(3):

    axs = fig.add_subplot(gs[i, 4])
    axs.imshow(
        ys[i].reshape(Nim, Nim),
        cmap="Reds",
        vmin=minval,
        vmax=maxval
        # norm=Normalize(vmin=0.0, clip=True),
    )
    axs.axis("off")


for i in range(3):
    y = jnp.exp(-1 * model.alpha[i][0] * (tv1.reshape(-1, 1)) ** 2) * g_samps[i][0]
    axs = fig.add_subplot(gs[i, 1])
    axs.plot(tv1, y[:, 0], c="red")
    # axs[0][i].set_ylim(-1, 1)
    axs.axis("off")


Nu = 1000
tu = jnp.linspace(-1.5, 1.5, Nu)
u_samp = model.sample_u_gp(tu, 1, jrnd.split(jrnd.PRNGKey(1), 2)).flatten()
u2d = map2matrix(lambda i, j: u_samp[i] * u_samp[j], jnp.arange(Nu), jnp.arange(Nu))
axs = fig.add_subplot(gs[1, 0])
axs.plot(tu, u_samp, c="blue")
axs.axis("off")

axs = fig.add_subplot(gs[1, 3])
# u2d = u2d * (u2d > 1e-5)
axs.imshow(
    u2d,
    cmap="Blues",
    interpolation="spline36",
    # norm=Normalize(vmin=0.0, clip=True),
)
axs.axis("off")

ty = jnp.linspace(-1.5, 1.5, 100)
y_samps = model.sample([ty] * 3, 1)
y1_samps = modelc1.sample([ty] * 3, 1)
fs_posl = [7, 14]
for i in range(3):

    axs = fig.add_subplot(gs[i, 2])
    axs.plot(ty, y1_samps[i], color="green")
    axs.axis("off")
    axs = fig.add_subplot(gs[i, 5])
    axs.plot(ty, y_samps[i] - y1_samps[i], color="green")
    axs.axis("off")
    axs = fig.add_subplot(gs[i, 6])
    axs.plot(ty, y_samps[i], color="green")
    axs.axis("off")

plt.savefig("sampling_diagram.pdf")
plt.show()

# %%

# %%

# %%
