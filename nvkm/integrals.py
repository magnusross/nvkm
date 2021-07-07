import operator
from functools import partial

import jax.numpy as jnp
from jax import config, jit, lax, vmap

from nvkm.utils import map_reduce, map2matrix

config.update("jax_enable_x64", True)


class Full:
    @staticmethod
    @jit
    def integ_1a(t, alpha, thet1, thet2, beta2):
        """
        Implements integral 1A from the supplementary material.
        """
        coeff = 0.5 * jnp.sqrt(jnp.pi / alpha)
        ea1 = lax.complex(-((thet1 + thet2) ** 2) / (4.0 * alpha), -beta2 - t * thet2)
        ea2 = lax.complex((thet1 * thet2) / alpha, 2 * (beta2 + t * thet2))
        return coeff * jnp.exp(ea1) * (1.0 + jnp.exp(ea2))

    @staticmethod
    @jit
    def integ_1b(t, alpha, thet1, p2, z2):
        """
        Implements integral 1B from the supplementary material.
        """
        coeff = jnp.sqrt(jnp.pi / (alpha + p2))
        ear = -(thet1 ** 2 + 4 * alpha * p2 * (t - z2) ** 2)
        eai = 4 * p2 * (t * thet1 - thet1 * z2)
        return coeff * jnp.exp(lax.complex(ear, eai) / (4 * (alpha + p2)))

    @staticmethod
    @jit
    def integ_2a(t, alpha, p1, z1, thet2, beta2):
        """
        Implements integral 2A from the supplementary material.
        """
        coeff = jnp.sqrt(jnp.pi / (alpha + p1))
        ea = -(4 * alpha * p1 * z1 ** 2 + thet2 ** 2) / (4 * (alpha + p1))
        ca = thet2 * (t - (p1 * z1) / (alpha + p1)) + beta2
        return coeff * jnp.exp(ea) * jnp.cos(ca)

    @staticmethod
    @jit
    def integ_2b(t, alpha, p1, z1, p2, z2):
        """
        Implements integral 2B from the supplementary material.
        """
        coeff = jnp.sqrt(jnp.pi / (alpha + p1 + p2))
        ea1 = alpha * (p1 * z1 ** 2 + p2 * (t - z2) ** 2)
        ea2 = p1 * p2 * (z1 + z2 - t) ** 2
        return coeff * jnp.exp(-(ea1 + ea2) / (alpha + p1 + p2))

    @classmethod
    def slow_I1(
        cls,
        t,
        zus,
        thetag,
        betag,
        thetus,
        betaus,
        wus,
        qus,
        sigg,
        sigu,
        alpha,
        pu,
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
                opa += wus[m] * cls.integ_1a(t, alpha, thetag[i], thetus[m], betaus[m])
                ona += wus[m] * cls.integ_1a(
                    t, alpha, -1.0 * thetag[i], thetus[m], betaus[m]
                )
            for n in range(Mu):
                opb += qus[n] * cls.integ_1b(t, alpha, thetag[i], pu, zus[n])
                onb += qus[n] * cls.integ_1b(t, alpha, -1.0 * thetag[i], pu, zus[n])
            o1 *= opa + sigu ** 2 * opb
            o2 *= ona + sigu ** 2 * onb
        out = 0.5 * jnp.real(jnp.exp(betag * 1j) * o1 + jnp.exp(-betag * 1j) * o2)
        return out

    @classmethod
    @partial(jit, static_argnums=(0,))
    def fast_I1(
        cls,
        t,
        zus,
        thetag,
        betag,
        thetus,
        betaus,
        wus,
        qus,
        sigg,
        sigu,
        alpha,
        pu,
    ):
        """
        Fast implementation of integral 1 from supplementary material.
        """
        o = vmap(
            lambda thetgij: map_reduce(
                lambda wui, thetui, betaui: wui
                * cls.integ_1a(t, alpha, thetgij, thetui, betaui),
                wus,
                thetus,
                betaus,
            )
            + sigu ** 2
            * map_reduce(
                lambda qui, zui: qui * cls.integ_1b(t, alpha, thetgij, pu, zui),
                qus,
                zus,
            )
        )(thetag)

        o1 = jnp.prod(o)
        return jnp.abs(o1) * jnp.cos(jnp.angle(o1) + betag)

    @classmethod
    def slow_I2(cls, t, zg, zus, thetus, betaus, wus, qus, sigg, sigu, alpha, pg, pu):
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
                os1 += wus[m] * cls.integ_2a(t, alpha, pg, zg[i], thetus[m], betaus[m])
            for n in range(Mu):
                os2 += qus[n] * cls.integ_2b(t, alpha, pg, zg[i], pu, zus[n])

            o *= os1 + sigu ** 2 * os2
        return sigg ** 2 * o

    @classmethod
    @partial(jit, static_argnums=(0,))
    def fast_I2(cls, t, zg, zus, thetus, betaus, wus, qus, sigg, sigu, alpha, pg, pu):
        """
        Fast implementation of integral 2 from supplementary material.
        """
        o1 = vmap(
            lambda zgij: map_reduce(
                lambda wi, thetui, betaui: wi
                * cls.integ_2a(t, alpha, pg, zgij, thetui, betaui),
                wus,
                thetus,
                betaus,
            )
        )(zg)

        o2 = vmap(
            lambda zgij: sigu ** 2
            * map_reduce(
                lambda qi, zui: qi * cls.integ_2b(t, alpha, pg, zgij, pu, zui),
                qus,
                zus,
            )
        )(zg)

        return sigg ** 2 * jnp.prod((o1 + o2))

    @classmethod
    def slow_I(
        cls,
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
            out += wgs[i] * cls.slow_I1(
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
            out += qgs[j] * cls.slow_I2(
                t,
                zgs[j],
                zus,
                thetus,
                betaus,
                wus,
                qus,
                sigg,
                sigu,
                alpha,
                pg,
                pu,
            )
        return out

    @classmethod
    @partial(jit, static_argnums=(0,))
    def fast_I(
        cls,
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
            * cls.fast_I1(
                t,
                zus,
                thetagi,
                betagi,
                thetus,
                betaus,
                wus,
                qus,
                sigg,
                sigu,
                alpha,
                pu,
            )
        )(
            thetags,
            betags,
            wgs,
        )

        o2 = vmap(
            lambda zgi, qgi: qgi
            * cls.fast_I2(
                t,
                zgi,
                zus,
                thetus,
                betaus,
                wus,
                qus,
                sigg,
                sigu,
                alpha,
                pg,
                pu,
            )
        )(
            zgs,
            qgs,
        )

        return jnp.sum(o1) + jnp.sum(o2)

    @classmethod
    @partial(jit, static_argnums=(0,))
    def I(
        cls,
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
                lambda thetags, betags, thetaus, betaus, wgs, qgs, wus, qus: cls.fast_I(
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
        )(
            ts,
        )


class Separable:
    @staticmethod
    @jit
    def I_phi_phi(t, alpha, thet1, thet2, beta1, beta2):
        coeff = 0.5 * jnp.sqrt(jnp.pi / alpha)
        ea1 = jnp.exp(-((thet1 + thet2) ** 2) / (4.0 * alpha))
        ea2 = jnp.cos(beta1 - beta2 - t * thet2) + jnp.exp(
            thet1 * thet2 / alpha
        ) * jnp.cos(beta1 + beta2 + t * thet2)
        return coeff * ea1 * ea2

    @staticmethod
    @jit
    def I_k_phi(t, alpha, p1, z1, thet2, beta2):
        coeff = jnp.sqrt(jnp.pi / (alpha + p1))
        ea1 = jnp.exp(-(4 * alpha * p1 * z1 ** 2 + thet2 ** 2) / (4 * (alpha + p1)))
        ea2 = jnp.cos(beta2 + thet2 * (t - (p1 * z1) / (alpha + p1)))
        return coeff * ea1 * ea2

    @staticmethod
    @jit
    def I_k_k(*args):
        return Full.integ_2b(*args)

    @classmethod
    @partial(jit, static_argnums=(0,))
    @partial(
        jnp.vectorize,
        excluded=(0, 1, 2, 3, 6, 7, 10, 11, 12, 13, 14),
        signature="(k),(k),(k),(k)->()",
    )
    def single_I(
        cls,
        t,
        zgs,
        zus,
        thetgs,
        betags,
        thetus,
        betaus,
        wgs,
        qgs,
        wus,
        qus,
        alpha,
        pg,
        pu,
    ):
        t1 = jnp.sum(
            vmap(
                lambda wgi, thetgi, betagi: vmap(
                    lambda wui, thetui, betaui: wgi
                    * wui
                    * cls.I_phi_phi(t, alpha, thetgi, thetui, betagi, betaui)
                )(wus, thetus, betaus)
            )(wgs, thetgs, betags)
        )

        t2 = jnp.sum(
            vmap(
                lambda qgi, zgi: vmap(
                    lambda wui, thetui, betaui: qgi
                    * wui
                    * cls.I_k_phi(t, alpha, pg, zgi, thetui, betaui)
                )(wus, thetus, betaus)
            )(qgs, zgs)
        )

        t3 = jnp.sum(
            vmap(
                lambda wgi, thetgi, betagi: vmap(
                    lambda qui, zui: qui
                    * wgi
                    * cls.I_k_phi(t, alpha, pu, zui, thetgi, betagi)
                )(qus, zus)
            )(wgs, thetgs, betags)
        )

        t4 = jnp.sum(
            vmap(
                lambda qgi, zgi: vmap(
                    lambda qui, zui: qui * qgi * cls.I_k_k(t, alpha, pg, zgi, pu, zui)
                )(qus, zus)
            )(qgs, zgs)
        )

        return t1 + t2 + t3 + t4

    @classmethod
    @partial(jit, static_argnums=(0,))
    def I(
        cls,
        ts,
        zgs,
        zus,
        athetagl,  # Ns x C x Nb
        abetagl,  # Ns x C x Nb
        thetaul,  # Ns x Nb
        betaul,  # Ns x Nb
        awgl,  # Ns x C x Nb
        aqgl,  # Ns x C x Nv
        wul,  # Ns x Nb
        qul,  # Ns x Nb
        alpha,
        pg,
        pu,
    ):

        return vmap(
            lambda ti: vmap(
                lambda athetags, abetags, thetaus, betaus, awgs, aqgs, wus, qus: jnp.prod(
                    cls.single_I(
                        ti,
                        zgs,
                        zus,
                        athetags,
                        abetags,
                        thetaus,
                        betaus,
                        awgs,
                        aqgs,
                        wus,
                        qus,
                        alpha,
                        pg,
                        pu,
                    )
                )
            )(athetagl, abetagl, thetaul, betaul, awgl, aqgl, wul, qul)
        )(
            ts,
        )


class Homogeneous(Separable):
    @classmethod
    @partial(jit, static_argnums=(0,))
    def I(
        cls,
        ts,
        zgs,
        zus,
        thetagl,  # Ns x Nb
        betagl,  # Ns x Nb
        thetaul,  # Ns x Nb
        betaul,  # Ns x Nb
        wgl,  # Ns x Nb
        qgl,  # Ns x Nv
        wul,  # Ns x Nb
        qul,  # Ns x Nb
        alpha,
        pg,
        pu,
    ):

        return vmap(
            lambda ti: vmap(
                lambda athetags, abetags, thetaus, betaus, awgs, aqgs, wus, qus: jnp.prod(
                    cls.single_I(
                        ti,
                        zgs,
                        zus,
                        athetags,
                        abetags,
                        thetaus,
                        betaus,
                        awgs,
                        aqgs,
                        wus,
                        qus,
                        alpha,
                        pg,
                        pu,
                    )
                )
            )(thetagl, betagl, thetaul, betaul, wgl, qgl, wul, qul)
        )(
            ts,
        )
