from functools import partial
from typing import List, Union

import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy as jsp
import matplotlib.pyplot as plt
from jax import jit, vmap

from .settings import JITTER
from .utils import l2p, map2matrix, eq_kernel
from .integrals import fast_I


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
    def compute_q(self, thetas, betas, ws):
        Phi = self.compute_Phi(thetas, betas)
        b = self.v - Phi @ ws
        return jsp.linalg.cho_solve((self.LKvv, True), b)

    def sample(self, t, Ns=100, key=jrnd.PRNGKey(1)):

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
            qs = vmap(lambda thi, bi, wi: self.compute_q(thi, bi, wi))(
                thetas, betas, ws
            )  # Ns x Nz
            kv = map2matrix(self.kernel, t, self.z, self.amp, self.ls)  # Nt x Nz
            kv = jnp.einsum("ij, kj", qs, kv)  # Nt x Ns

            samps += kv.T

        return samps


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

    def sample(self, t, N_s=100, key=jrnd.PRNGKey(1)):

        samples = jnp.zeros((len(t), N_s))

        skey = jrnd.split(key, 4)

        u_gp = self.u_gp
        thetaul = u_gp.sample_thetas(skey[0], (N_s, u_gp.N_basis, 1), u_gp.ls)
        betaul = u_gp.sample_betas(skey[1], (N_s, u_gp.N_basis))
        wul = u_gp.sample_ws(skey[2], (N_s, u_gp.N_basis))

        qul = vmap(lambda thi, bi, wi: u_gp.compute_q(thi, bi, wi))(
            thetaul.reshape(N_s, -1, 1), betaul, wul
        )

        for i in range(0, self.C):
            skey = jrnd.split(skey[3], 4)

            G_gp_i = self.g_gps[i]
            thetagl = G_gp_i.sample_thetas(
                skey[0], (N_s, G_gp_i.N_basis, G_gp_i.D), G_gp_i.ls
            )
            betagl = G_gp_i.sample_betas(skey[1], (N_s, G_gp_i.N_basis))
            wgl = G_gp_i.sample_ws(skey[2], (N_s, G_gp_i.N_basis))

            qgl = vmap(lambda thi, bi, wi: G_gp_i.compute_q(thi, bi, wi))(
                thetagl.reshape(N_s, -1, 1), betagl, wgl
            )
            print(G_gp_i.v.shape)
            print(u_gp.v.shape)

            print(
                thetagl.shape,
                betagl.shape,
                thetaul.shape,
                betaul.shape,
                wgl.shape,
                qgl.shape,
                wul.shape,
                qul.shape,
            )

            samples += vmap(
                lambda ti: vmap(
                    lambda thetags, betags, thetaus, betaus, wgs, qgs, wus, qus: fast_I(
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
                        G_gp_i.amp,
                        sigu=u_gp.amp,
                        alpha=self.alpha,
                        pg=l2p(G_gp_i.ls),
                        pu=l2p(u_gp.ls),
                    )
                )(
                    thetagl, betagl, thetaul, betaul, wgl, qgl, wul, qul,
                )
            )(t)

        return samples
