import jax.numpy as jnp

import matplotlib.pyplot as plt

from nvkm.models import EQApproxGP


t = jnp.linspace(-10, 10, 500)
gp1D = EQApproxGP(z=None, v=None, amp=1000, ls=0.5, noise=0.01, N_basis=1000)
samps = gp1D.sample(t, Ns=10)
plt.plot(t, samps)
plt.savefig("basegp.png")
plt.show()