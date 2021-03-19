from jax import jit
from jax import lax
import jax.numpy as jnp

from .utils import method


@jit
def integ_1a(t, alpha, thet1, beta1, thet2, beta2):
    coeff = 0.5 * jnp.sqrt(jnp.pi / alpha)
    ea1 = lax.complex(
        -((thet1 + thet2) ** 2) / (4.0 * alpha), beta1 - beta2 - t * thet2
    )
    ea2 = lax.complex((thet1 * thet2) / alpha, 2 * (beta2 + t * thet2))
    return coeff * jnp.exp(ea1) * (1.0 + jnp.exp(ea2))


@jit
def integ_1b(t, alpha, thet1, beta1, p2, z2):
    coeff = jnp.sqrt(jnp.pi / (alpha + p2))
    ear = -(thet1 ** 2 + 4 * alpha * p2 * (t - z2) ** 2)
    eai = 4 * (beta1 * alpha + p2 * (beta1 + t * thet1 - thet1 * z2))
    return coeff * jnp.exp(lax.complex(ear, eai) / (4 * (alpha + p2)))


@jit
def integ_2a(t, alpha, p1, z1, thet2, beta2):
    coeff = jnp.sqrt(jnp.pi / (alpha + p1))
    ea = -(4 * alpha * p1 * z1 ** 2 + thet2 ** 2) / (4 * (alpha + p1))
    ca = thet2 * (t - (p1 * z1) / (alpha + p1)) + beta2
    return coeff * jnp.exp(ea) * jnp.cos(ca)


@jit
def integ_2b(t, alpha, p1, z1, p2, z2):
    coeff = jnp.sqrt(jnp.pi / (alpha + p1 + p2))
    ea1 = alpha * (p1 * z1 ** 2 + p2 * (t - z2) ** 2)
    ea2 = p1 * p2 * (z1 + z2 - t) ** 2
    return coeff * jnp.exp(-(ea1 + ea2) / (alpha + p1 + p2))

