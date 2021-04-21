import jax.numpy as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt
import nvkm.utils
from nvkm.models import EQApproxGP


t = jnp.linspace(-5, 5, 500)
Nvg = 10
tf = jnp.linspace(-5, 5, int(jnp.sqrt(Nvg)))
tm2 = jnp.meshgrid(tf, tf)
z = jnp.vstack((tm2[0].flatten(), tm2[1].flatten())).T

amp = 0.1
v = (
    EQApproxGP(z=None, v=None, amp=amp, D=2, ls=0.5, noise=0.00, N_basis=1000)
    .sample(z, 1)
    .flatten()
)
gp1D = EQApproxGP(z=z, v=v, amp=amp, D=2, ls=0.5, noise=0.00, N_basis=1000)

samps = gp1D.sample(jnp.vstack((t, t)).T, Ns=10)
plt.plot(t, samps)
plt.show()
#%%
m, K = nvkm.utils.exact_gp_posterior(nvkm.utils.eq_kernel, t, z, v, amp, 0.5)
s = jrnd.multivariate_normal(jrnd.PRNGKey(1000), m, K + jnp.eye(len(t)) * 1e-7, (10,)).T
plt.plot(t, m)
plt.plot(t, m + jnp.sqrt(jnp.diag(K)))
plt.plot(t, m - jnp.sqrt(jnp.diag(K)))
plt.show()

# %%
