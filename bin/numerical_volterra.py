# %%
import numpy as onp
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt

# %%
def G1(x, alpha=0.2, delta=0.0):
    return onp.exp(-alpha * x ** 2) * (onp.sin(2 * x) + delta * (onp.sin(x - 0.1)) ** 3)


def G2(x1, x2, alpha=0.2, delta=0.0):
    return onp.exp(-alpha * (x1 ** 2 + x2 ** 2)) * (
        onp.cos(0.5 * x1 + 1.0 * x2) ** 2 + delta * (onp.sin(x1 + x2 - 0.1)) ** 3
    )


#%%
delta = 0.3
tg = onp.linspace(-6, 6, 100)
plt.plot(tg, G1(tg))
plt.plot(tg, G1(tg, delta=delta))
plt.savefig("G1.png")
plt.show()


fig, axs = plt.subplots(2, 1)
axs[0].plot(tg, G2(tg, tg))
axs[0].plot(tg, G2(tg, tg, delta=delta))
axs[1].plot(tg, G2(tg, -tg))
axs[1].plot(tg, G2(tg, -tg, delta=delta))
plt.savefig("G2.png")
plt.show()
# %%
def u(t, delta=0.1):
    return onp.sin(4 * t) + onp.cos(2 * t) + onp.sin(t) ** 2 * delta


t = onp.linspace(-3, 3, 30)
plt.plot(t, u(t))
plt.plot(t, u(t, delta=delta))
plt.savefig("input.png")

plt.show()
# %%


def volt(
    t, delta=0.0,
):
    c1 = quad(
        lambda tau: G1(t - tau, delta=delta) * u(tau, delta=delta), -onp.inf, onp.inf
    )
    c2 = dblquad(
        lambda tau1, tau2: G2(t - tau1, t - tau2, delta=delta)
        * u(tau1, delta=delta)
        * u(tau2, delta=delta),
        -onp.inf,
        onp.inf,
        -onp.inf,
        onp.inf,
    )
    print(t)
    return c1[0] + c2[0]


y1 = [volt(ti, delta=delta) for ti in t]
y2 = [volt(ti) for ti in t]

# %%
plt.plot(t, y1)
plt.plot(t, y2)
plt.savefig("output.png")
# %%
