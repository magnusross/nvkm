from jax import jit
from jax import lax
import jax.numpy as jnp


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


def slow_I1(
    t,
    zus,
    thetag,
    betag,
    thetus,
    betaus,
    wus,
    qus,
    sigg=1.0,
    sigu=1.0,
    alpha=1.0,
    pu=1.0,
):
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
            opa += wus[m] * integ_1a(t, alpha, thetag[i], betag, thetus[m], betaus[m])
            ona += wus[m] * integ_1a(
                t, alpha, -1.0 * thetag[i], -1.0 * betag, thetus[m], betaus[m]
            )
        for n in range(Mu):
            opb += qus[n] * integ_1b(t, alpha, thetag[i], betag, pu, zus[n])
            onb += qus[n] * integ_1b(
                t, alpha, -1.0 * thetag[i], -1.0 * betag, pu, zus[n]
            )
        o1 *= jnp.sqrt(2.0 / Nl) * opa + sigu ** 2 * opb
        o2 *= jnp.sqrt(2.0 / Nl) * ona + sigu ** 2 * onb

        return 0.5 * jnp.sqrt(2.0 / Nl) * (o1 + o2)


def slow_I2(
    t, zg, zus, thetus, betaus, wus, qus, sigg=1.0, sigu=1.0, alpha=1.0, pg=1.0, pu=1.0
):
    c = zg.shape[0]  # order of the term
    Nl = wus.shape[0]  # number of basis functions
    Mu = zus.shape[0]  # number of u inducing points

    o1 = 1.0
    o2 = 1.0

    for i in range(c):
        os1 = 0.0
        os2 = 0.0

        for m in range(Nl):
            os1 += wus[m] * integ_2a(t, alpha, pg, zg[i], thetus[m], betaus[m])
        for n in range(Mu):
            os2 += qus[n] * integ_2b(t, alpha, pg, zg[i], pu, zus[n])

        o1 *= jnp.sqrt(2 / Nl) * os1
        o2 *= sigu * os2
    return sigg * (o1 + o2)


def fast_I2(
    t, zg, zus, thetus, betaus, wus, qus, sigg=1.0, sigu=1.0, alpha=1.0, pg=1.0, pu=1.0
):
    c = zg.shape[0]  # order of the term
    Nl = wus.shape[0]  # number of basis functions
    Mu = zus.shape[0]  # number of u inducing points

    o1 = 1.0
    o2 = 1.0

    for i in range(c):
        os1 = 0.0
        os2 = 0.0

        for m in range(Nl):
            os1 += wus[m] * integ_2a(t, alpha, pg, zg[i], thetus[m], betaus[m])
        for n in range(Mu):
            os2 += qus[n] * integ_2b(t, alpha, pg, zg[i], pu, zus[n])

        o1 *= jnp.sqrt(2 / Nl) * os1
        o2 *= sigu * os2
    return sigg * (o1 + o2)


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
    sigg=1.0,
    sigu=1.0,
    alpha=1.0,
    pg=1.0,
    pu=1.0,
):
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
            sigg=sigg,
            sigu=sigu,
            alpha=alpha,
            pu=pu,
        )
    for j in range(Mg):
        out += qgs[j] * slow_I2(
            t,
            zgs[j],
            zus,
            thetus,
            betaus,
            wus,
            qus,
            sigg=sigg,
            sigu=sigu,
            alpha=alpha,
            pu=pu,
        )

    return out
