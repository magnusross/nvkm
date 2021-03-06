from jax import jit, vmap
import jax.numpy as jnp
from jax import lax
from jax import config
from nvkm.utils import map_reduce
import operator

config.update("jax_enable_x64", True)


@jit
def integ_1a(t, alpha, thet1, thet2, beta2):
    """
    Implements integral 1A from the supplementary material.
    """
    coeff = 0.5 * jnp.sqrt(jnp.pi / alpha)
    ea1 = lax.complex(-((thet1 + thet2) ** 2) / (4.0 * alpha), -beta2 - t * thet2)
    ea2 = lax.complex((thet1 * thet2) / alpha, 2 * (beta2 + t * thet2))
    return coeff * jnp.exp(ea1) * (1.0 + jnp.exp(ea2))


@jit
def integ_1b(t, alpha, thet1, p2, z2):
    """
    Implements integral 1B from the supplementary material.
    """
    coeff = jnp.sqrt(jnp.pi / (alpha + p2))
    ear = -(thet1 ** 2 + 4 * alpha * p2 * (t - z2) ** 2)
    eai = 4 * p2 * (t * thet1 - thet1 * z2)
    return coeff * jnp.exp(lax.complex(ear, eai) / (4 * (alpha + p2)))


@jit
def integ_2a(t, alpha, p1, z1, thet2, beta2):
    """
    Implements integral 2A from the supplementary material.
    """
    coeff = jnp.sqrt(jnp.pi / (alpha + p1))
    ea = -(4 * alpha * p1 * z1 ** 2 + thet2 ** 2) / (4 * (alpha + p1))
    ca = thet2 * (t - (p1 * z1) / (alpha + p1)) + beta2
    return coeff * jnp.exp(ea) * jnp.cos(ca)


@jit
def integ_2b(t, alpha, p1, z1, p2, z2):
    """
    Implements integral 2B from the supplementary material.
    """
    coeff = jnp.sqrt(jnp.pi / (alpha + p1 + p2))
    ea1 = alpha * (p1 * z1 ** 2 + p2 * (t - z2) ** 2)
    ea2 = p1 * p2 * (z1 + z2 - t) ** 2
    return coeff * jnp.exp(-(ea1 + ea2) / (alpha + p1 + p2))


def slow_I1(
    t, zus, thetag, betag, thetus, betaus, wus, qus, sigg, sigu, alpha, pu,
):
    """
    Slow implementation of integral 1 from supplementary material for testing.
    """
    c = thetag.shape[0]  # order of the term
    Nl = wus.shape[0]  # number of basis functions
    Mu = zus.shape[0]

    o1 = 1.0
    o2 = 1.0
    for i in range(c):
        opa = 0.0
        opb = 0.0
        ona = 0.0
        onb = 0.0
        for m in range(Nl):
            opa += wus[m] * integ_1a(t, alpha, thetag[i], thetus[m], betaus[m])
            ona += wus[m] * integ_1a(t, alpha, -1.0 * thetag[i], thetus[m], betaus[m])
        for n in range(Mu):
            opb += qus[n] * integ_1b(t, alpha, thetag[i], pu, zus[n])
            onb += qus[n] * integ_1b(t, alpha, -1.0 * thetag[i], pu, zus[n])
        o1 *= opa + sigu ** 2 * opb
        o2 *= ona + sigu ** 2 * onb
    out = 0.5 * jnp.real(jnp.exp(betag * 1j) * o1 + jnp.exp(-betag * 1j) * o2)
    return out


@jit
def fast_I1(
    t, zus, thetag, betag, thetus, betaus, wus, qus, sigg, sigu, alpha, pu,
):
    """
    Fast implementation of integral 1 from supplementary material.
    """
    o = vmap(
        lambda thetgij: map_reduce(
            lambda wui, thetui, betaui: wui
            * integ_1a(t, alpha, thetgij, thetui, betaui),
            wus,
            thetus,
            betaus,
        )
        + sigu ** 2
        * map_reduce(
            lambda qui, zui: qui * integ_1b(t, alpha, thetgij, pu, zui), qus, zus,
        )
    )(thetag)

    o1 = jnp.prod(o)
    return jnp.abs(o1) * jnp.cos(jnp.angle(o1) + betag)


def slow_I2(t, zg, zus, thetus, betaus, wus, qus, sigg, sigu, alpha, pg, pu):
    """
    Slow implementation of integral 2 from supplementary material for testing.
    """
    c = zg.shape[0]  # order of the term
    Nl = wus.shape[0]  # number of basis functions
    Mu = zus.shape[0]  # number of u inducing points

    o = 1.0

    for i in range(c):
        os1 = 0.0
        os2 = 0.0

        for m in range(Nl):
            os1 += wus[m] * integ_2a(t, alpha, pg, zg[i], thetus[m], betaus[m])
        for n in range(Mu):
            os2 += qus[n] * integ_2b(t, alpha, pg, zg[i], pu, zus[n])

        o *= os1 + sigu ** 2 * os2
    return sigg ** 2 * o


@jit
def fast_I2(t, zg, zus, thetus, betaus, wus, qus, sigg, sigu, alpha, pg, pu):
    """
    Fast implementation of integral 2 from supplementary material.
    """
    o1 = vmap(
        lambda zgij: map_reduce(
            lambda wi, thetui, betaui: wi
            * integ_2a(t, alpha, pg, zgij, thetui, betaui),
            wus,
            thetus,
            betaus,
        )
    )(zg)

    o2 = vmap(
        lambda zgij: sigu ** 2
        * map_reduce(
            lambda qi, zui: qi * integ_2b(t, alpha, pg, zgij, pu, zui), qus, zus,
        )
    )(zg)

    return sigg ** 2 * jnp.prod((o1 + o2))


def slow_I(
    t,
    zgs,
    zus,
    thetags,
    betags,
    thetus,
    betaus,
    wgs,
    qgs,
    wus,
    qus,
    sigg,
    sigu=1.0,
    alpha=1.0,
    pg=1.0,
    pu=1.0,
):
    """
    Slow implementation of Eqn 5 for testing.
    """
    Nl = wgs.shape[0]
    Mg = zgs.shape[0]

    out = 0.0
    for i in range(Nl):
        out += wgs[i] * slow_I1(
            t,
            zus,
            thetags[i],
            betags[i],
            thetus,
            betaus,
            wus,
            qus,
            sigg,
            sigu,
            alpha,
            pu,
        )
    for j in range(Mg):
        out += qgs[j] * slow_I2(
            t, zgs[j], zus, thetus, betaus, wus, qus, sigg, sigu, alpha, pg, pu,
        )
    return out


@jit
def fast_I(
    t,
    zgs,
    zus,
    thetags,
    betags,
    thetus,
    betaus,
    wgs,
    qgs,
    wus,
    qus,
    sigg,
    sigu,
    alpha,
    pg,
    pu,
):
    """
    Fast implementation of Eqn 5.
    """

    o1 = vmap(
        lambda thetagi, betagi, wgi,: wgi
        * fast_I1(
            t, zus, thetagi, betagi, thetus, betaus, wus, qus, sigg, sigu, alpha, pu,
        )
    )(thetags, betags, wgs,)

    o2 = vmap(
        lambda zgi, qgi: qgi
        * fast_I2(t, zgi, zus, thetus, betaus, wus, qus, sigg, sigu, alpha, pg, pu,)
    )(zgs, qgs,)

    return jnp.sum(o1) + jnp.sum(o2)


@jit
def map_fast_I(
    ts,
    zgs,
    zus,
    thetagl,
    betagl,
    thetaul,
    betaul,
    wgl,
    qgl,
    wul,
    qul,
    ampg,
    ampu,
    alpha,
    pg,
    pu,
):
    return vmap(
        lambda ti: vmap(
            lambda thetags, betags, thetaus, betaus, wgs, qgs, wus, qus: fast_I(
                ti,
                zgs,
                zus,
                thetags,
                betags,
                thetaus,
                betaus,
                wgs,
                qgs,
                wus,
                qus,
                ampg,
                ampu,
                alpha,
                pg,
                pu,
            )
        )(thetagl, betagl, thetaul, betaul, wgl, qgl, wul, qul)
    )(ts,)

