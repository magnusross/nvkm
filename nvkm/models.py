from functools import partial
from typing import List, Union, Tuple, Callable

import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy as jsp
import matplotlib.pyplot as plt
from jax import jit, vmap

from .settings import JITTER
from .utils import l2p, map2matrix, eq_kernel
from .integrals import fast_I
from .vi import (
    VariationalDistribution,
    VIPars,
    IndependentGaussians,
    gaussain_likelihood,
)


class EQApproxGP:
    def __init__(
        self,
        z: Union[jnp.DeviceArray, None] = None,
        v: Union[jnp.DeviceArray, None] = None,
        N_basis: int = 500,
        D: int = 1,
        ls: float = 1.0,
        amp: float = 1.0,
        noise: float = 0.0,
    ):

        self.z = z
        self.v = v
        self.N_basis = N_basis
        self.D = D

        self.ls = ls
        self.pr = l2p(ls)
        self.amp = amp
        self.noise = noise

        self.Kvv = None
        self.LKvv = None

        if self.z == None:
            pass

        else:
            try:
                assert self.z.shape[1] == self.D

            except IndexError:
                self.z = self.z.reshape(-1, 1)
                assert self.D == 1

            except AssertionError:
                raise ValueError(
                    "Dimension of inducing points does not match dimension of GP."
                )

            self.Kvv, self.LKvv = self.compute_covariances(amp, ls)

    @partial(jit, static_argnums=(0,))
    def compute_covariances(self, amp, ls):
        Kvv = map2matrix(self.kernel, self.z, self.z, amp, ls) + (
            self.noise + JITTER
        ) * jnp.eye(self.z.shape[0])
        LKvv = jnp.linalg.cholesky(Kvv)
        return Kvv, LKvv

    @partial(jit, static_argnums=(0,))
    def fast_covariance_recompute(self, new_amp):
        factor = new_amp ** 2 / self.amp ** 2
        return factor * self.Kvv, factor * self.LKvv

    @partial(jit, static_argnums=(0,))
    def compute_Phi(self, thetas, betas):
        return vmap(lambda zi: self.phi(zi, thetas, betas))(self.z)

    @partial(jit, static_argnums=(0,))
    def kernel(self, t, tp, amp, ls):
        return eq_kernel(t, tp, amp, ls)

    @partial(jit, static_argnums=(0, 2))
    def sample_thetas(self, key, shape, ls):
        # FT of isotropic gaussain is inverser varience
        return jrnd.normal(key, shape) / ls

    @partial(jit, static_argnums=(0, 2))
    def sample_betas(self, key, shape):
        return jrnd.uniform(key, shape, maxval=2 * jnp.pi)

    @partial(jit, static_argnums=(0, 2))
    def sample_ws(self, key, shape):
        return jrnd.normal(key, shape)

    @partial(jit, static_argnums=(0,))
    def phi(self, t, theta, beta):
        return jnp.sqrt(2 / self.N_basis) * jnp.cos(jnp.dot(theta, t) + beta)

    @partial(jit, static_argnums=(0,))
    def compute_q(self, v, LKvv, thetas, betas, ws):
        Phi = self.compute_Phi(thetas, betas)
        b = v - Phi @ ws
        return jsp.linalg.cho_solve((LKvv, True), b)

    @partial(jit, static_argnums=(0,))
    def _sample(self, t, v, amp, Ns=100, key=jrnd.PRNGKey(1)):

        try:
            assert t.shape[1] == self.D

        except IndexError:
            t = t.reshape(-1, 1)
            assert self.D == 1

        except AssertionError:
            raise ValueError("Dimension of input does not match dimension of GP.")
        # sample random parameters
        skey = jrnd.split(key, 3)
        thetas = self.sample_thetas(skey[0], (Ns, self.N_basis, self.D), self.ls)
        betas = self.sample_betas(skey[1], (Ns, self.N_basis))
        ws = self.sample_ws(skey[2], (Ns, self.N_basis))

        # fourier basis part
        samps = vmap(
            lambda ti: vmap(lambda thi, bi, wi: jnp.dot(wi, self.phi(ti, thi, bi)))(
                thetas, betas, ws
            )
        )(t)

        # canonical basis part
        if self.z == None:
            pass
        else:
            qs = vmap(lambda thi, bi, wi: self.compute_q(v, self.LKvv, thi, bi, wi))(
                thetas, betas, ws
            )  # Ns x Nz
            kv = map2matrix(self.kernel, t, self.z, amp, self.ls)  # Nt x Nz
            kv = jnp.einsum("ij, kj", qs, kv)  # Nt x Ns

            samps += kv.T

        return samps

    def sample(self, t, Ns=100, key=jrnd.PRNGKey(1)):
        return self._sample(t, self.v, self.amp, Ns=Ns, key=key)


class NVKM:
    def __init__(
        self,
        zgs: List[Union[jnp.DeviceArray, None]] = [None],
        vgs: List[Union[jnp.DeviceArray, None]] = [None],
        zu: Union[jnp.DeviceArray, None] = None,
        vu: Union[jnp.DeviceArray, None] = None,
        N_basis: int = 500,
        C: int = 1,
        noise: float = 0.5,
        alpha: float = 1.0,
        lsgs: List[float] = [1.0],
        ampgs: List[float] = [1.0],
        lsu: float = 1.0,
        ampu: float = 1.0,
    ):
        self.zgs = zgs
        self.vgs = vgs
        self.zu = zu
        self.vu = vu

        self.N_basis = N_basis
        self.C = C
        self.noise = noise
        self.alpha = alpha

        self.lsgs = lsgs
        self.lsu = lsu
        self.ampgs = ampgs
        self.ampu = ampu

        self.g_gps = self.set_G_gps(ampgs, lsgs)
        self.u_gp = self.set_u_gp(ampu, lsu)

    def set_G_gps(self, ampgs, lsgs):
        gps = [
            EQApproxGP(
                z=self.zgs[i],
                v=self.vgs[i],
                N_basis=self.N_basis,
                D=i + 1,
                ls=lsgs[i],
                amp=ampgs[i],
            )
            for i in range(self.C)
        ]
        return gps

    def set_u_gp(self, ampu, lsu):
        return EQApproxGP(
            z=self.zu, v=self.vu, N_basis=self.N_basis, D=1, ls=lsu, amp=ampu
        )

    def _sample(self, t, vgs, vu, ampgs, N_s=10, key=jrnd.PRNGKey(1)):

        samps = jnp.zeros((len(t), N_s))
        skey = jrnd.split(key, 4)

        u_gp = self.u_gp
        thetaul = u_gp.sample_thetas(skey[0], (N_s, u_gp.N_basis, 1), u_gp.ls)
        betaul = u_gp.sample_betas(skey[1], (N_s, u_gp.N_basis))
        wul = u_gp.sample_ws(skey[2], (N_s, u_gp.N_basis))

        qul = vmap(lambda thi, bi, wi: u_gp.compute_q(vu, u_gp.LKvv, thi, bi, wi))(
            thetaul, betaul, wul
        )

        for i in range(0, self.C):
            skey = jrnd.split(skey[3], 4)

            G_gp_i = self.g_gps[i]
            thetagl = G_gp_i.sample_thetas(
                skey[0], (N_s, G_gp_i.N_basis, G_gp_i.D), G_gp_i.ls
            )
            betagl = G_gp_i.sample_betas(skey[1], (N_s, G_gp_i.N_basis))
            wgl = G_gp_i.sample_ws(skey[2], (N_s, G_gp_i.N_basis))

            _, G_LKvv = G_gp_i.fast_covariance_recompute(ampgs[i])
            qgl = vmap(
                lambda thi, bi, wi: G_gp_i.compute_q(vgs[i], G_LKvv, thi, bi, wi)
            )(thetagl, betagl, wgl)

            samps += vmap(
                lambda thetags, betags, thetaus, betaus, wgs, qgs, wus, qus: vmap(
                    lambda ti: fast_I(
                        ti,
                        G_gp_i.z,
                        u_gp.z,
                        thetags,
                        betags,
                        thetaus,
                        betaus,
                        wgs,
                        qgs,
                        wus,
                        qus,
                        ampgs[i],
                        sigu=u_gp.amp,
                        alpha=self.alpha,
                        pg=G_gp_i.pr,
                        pu=u_gp.pr,
                    )
                )(t)
            )(thetagl, betagl, thetaul, betaul, wgl, qgl, wul, qul,).T

        return samps

    def sample(self, t, N_s=10, key=jrnd.PRNGKey(1)):
        return self._sample(t, self.vgs, self.vu, self.ampgs, N_s=N_s, key=key)


class VariationalNVKM(NVKM):
    def __init__(
        self,
        zgs: List[jnp.DeviceArray],
        zu: jnp.DeviceArray,
        data: Tuple[jnp.DeviceArray, jnp.DeviceArray],
        q_class: VariationalDistribution,
        q_pars_init: Union[VIPars, None] = None,
        liklihood: Callable = gaussain_likelihood,
        N_basis: int = 500,
        C: int = 1,
        ampgs_init: List[float] = [1.0],
        noise_init: float = 0.5,
        alpha: float = 1.0,
        lsgs: List[float] = [1.0],
        lsu: float = 1.0,
        ampu: float = 1.0,
    ):
        vgs = [None] * C
        vu = None

        super().__init__(
            zgs=zgs,
            zu=zu,
            vgs=vgs,
            vu=vu,
            N_basis=N_basis,
            C=C,
            ampgs=ampgs_init,
            noise=noise_init,
            alpha=alpha,
            lsgs=lsgs,
            lsu=lsu,
            ampu=ampu,
        )

        if q_pars_init == None:
            self.q_of_v = q_class().initialize(data)
        else:
            self.q_of_v = q_class(init_pars=q_pars_init)

    def _var_sample(self, t, q_pars, ampgs, N_s=10, key=jrnd.PRNGKey(1)):

        v_samps = self.q_of_v._sample(q_pars, N_s, key)
        # samps = self._sample(t, v_samps["gs"], v_samps["u"], ampgs, N_s)

        skey = jrnd.split(key, 4)
        u_gp = self.u_gp
        thetaul = u_gp.sample_thetas(skey[0], (N_s, u_gp.N_basis, 1), u_gp.ls)
        betaul = u_gp.sample_betas(skey[1], (N_s, u_gp.N_basis))
        wul = u_gp.sample_ws(skey[2], (N_s, u_gp.N_basis))

        qul = vmap(
            lambda vui, thi, bi, wi: u_gp.compute_q(vui, u_gp.LKvv, thi, bi, wi)
        )(v_samps["u"], thetaul, betaul, wul)

        samps = jnp.zeros((len(t), N_s))
        for i in range(0, self.C):
            skey = jrnd.split(skey[3], 4)

            G_gp_i = self.g_gps[i]
            thetagl = G_gp_i.sample_thetas(
                skey[0], (N_s, G_gp_i.N_basis, G_gp_i.D), G_gp_i.ls
            )
            betagl = G_gp_i.sample_betas(skey[1], (N_s, G_gp_i.N_basis))
            wgl = G_gp_i.sample_ws(skey[2], (N_s, G_gp_i.N_basis))

            # print(G_gp_i.LKvv, G_Lkvv)
            _, G_LKvv = G_gp_i.fast_covariance_recompute(ampgs[i])
            qgl = vmap(
                lambda vgi, thi, bi, wi: G_gp_i.compute_q(vgi, G_LKvv, thi, bi, wi)
            )(v_samps["gs"][i], thetagl, betagl, wgl)
            # samps += jnp.zeros((len(t), N_s))
            samps += vmap(
                lambda thetags, betags, thetaus, betaus, wgs, qgs, wus, qus: vmap(
                    lambda ti: fast_I(
                        ti,
                        G_gp_i.z,
                        u_gp.z,
                        thetags,
                        betags,
                        thetaus,
                        betaus,
                        wgs,
                        qgs,
                        wus,
                        qus,
                        ampgs[i],
                        sigu=u_gp.amp,
                        alpha=self.alpha,
                        pg=G_gp_i.pr,
                        pu=u_gp.pr,
                    )
                )(t)
            )(thetagl, betagl, thetaul, betaul, wgl, qgl, wul, qul,).T

        return samps
