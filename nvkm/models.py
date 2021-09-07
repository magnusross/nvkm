import logging
import pickle
from functools import partial
from typing import Callable, List, Tuple, Union, Dict

import jax.experimental.optimizers as opt
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy as jsp
import matplotlib.pyplot as plt
from jax import jit, value_and_grad, vmap
from jax.config import config
from jax.experimental.host_callback import id_print

from .integrals import Full, Homogeneous, Separable, CausalSeparable, CausalHomogeneous
from .settings import JITTER
from .utils import choleskyize, eq_kernel, l2p, map2matrix
from .vi import (
    MOIndependentGaussians,
    gaussian_likelihood,
)

config.update("jax_enable_x64", True)


class EQApproxGP:
    def __init__(
        self,
        z: Union[jnp.ndarray, None] = None,
        v: Union[jnp.ndarray, None] = None,
        N_basis: int = 500,
        D: int = 1,
        ls: float = 1.0,
        amp: float = 1.0,
        noise: float = 0.0,
    ):
        """
        Implements the functional sampling method from
        "Efficiently Sampling Functions from Gaussian Process Posteriors"  by
        Wilson et al. for a GP with EQ (SE) kernel.

        Args:
            z (Union[jnp.ndarray, None], optional): Inducing input locations. Defaults to None.
            v (Union[jnp.ndarray, None], optional): Inducing ouputs. Defaults to None.
            N_basis (int, optional): Number of basis functions in approximation. Defaults to 500.
            D (int, optional): Input dimension. Defaults to 1.
            ls (float, optional): lengthscale. Defaults to 1.0.
            amp (float, optional): amplitude. Defaults to 1.0.
            noise (float, optional): noise standard deviation. Defaults to 0.0.

        Raises:
            ValueError: if inducing inputs dimension and GP dimension don't match
        """

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

        if self.z is None:
            pass
        else:
            self.Kvv, self.LKvv = self.compute_covariances(amp, ls)

    @partial(jit, static_argnums=(0,))
    def compute_covariances(self, amp: float, ls: float) -> Tuple[jnp.ndarray]:
        Kvv = map2matrix(self.kernel, self.z, self.z, amp, ls) + (
            self.noise + JITTER
        ) * jnp.eye(self.z.shape[0])
        LKvv = jnp.linalg.cholesky(Kvv)
        return Kvv, LKvv

    @partial(jit, static_argnums=(0,))
    def kernel(
        self, t: jnp.ndarray, tp: jnp.ndarray, amp: float, ls: float
    ) -> jnp.ndarray:
        return eq_kernel(t, tp, amp, ls)

    @partial(jit, static_argnums=(0, 2))
    def sample_thetas(self, key: jrnd.PRNGKey, shape: tuple, ls: float) -> jnp.ndarray:
        # FT of isotropic gaussain is inverse varience
        return jrnd.normal(key, shape) / ls

    @partial(jit, static_argnums=(0, 2))
    def sample_betas(self, key: jrnd.PRNGKey, shape: tuple) -> jnp.ndarray:
        return jrnd.uniform(key, shape, maxval=2 * jnp.pi)

    @partial(jit, static_argnums=(0, 2))
    def sample_ws(self, key: jrnd.PRNGKey, shape: tuple, amp: float) -> jnp.ndarray:
        return amp * jnp.sqrt(2 / self.N_basis) * jrnd.normal(key, shape)

    @partial(jit, static_argnums=(0,))
    def phi(self, t: jnp.ndarray, theta: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
        o = jnp.cos(jnp.dot(theta, t) + beta)
        return o

    @partial(jit, static_argnums=(0,))
    def compute_Phi(self, thetas: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
        return vmap(lambda zi: self.phi(zi, thetas, betas))(self.z)

    @partial(jit, static_argnums=(0,))
    def compute_q(
        self,
        v: jnp.ndarray,
        LKvv: jnp.ndarray,
        thetas: jnp.ndarray,
        betas: jnp.ndarray,
        ws: jnp.ndarray,
    ) -> jnp.ndarray:
        # print(thetas.shape, betas.shape)
        Phi = self.compute_Phi(thetas, betas)
        b = v - Phi @ ws
        # print(b.shape)
        return jsp.linalg.cho_solve((LKvv, True), b)

    @partial(jit, static_argnums=(0, 2))
    def sample_basis(
        self, key: jrnd.PRNGKey, Ns: int, amp: float, ls: float
    ) -> Tuple[jnp.ndarray]:
        skey = jrnd.split(key, 3)
        thetas = self.sample_thetas(skey[0], (Ns, self.N_basis, self.D), ls)
        betas = self.sample_betas(skey[1], (Ns, self.N_basis))
        ws = self.sample_ws(skey[2], (Ns, self.N_basis), amp)
        return thetas, betas, ws

    @partial(jit, static_argnums=(0, 5))
    def _sample(
        self,
        t: jnp.ndarray,
        vs: jnp.ndarray,
        amp: float,
        ls: float,
        Ns: int,
        key: jrnd.PRNGKey,
    ) -> jnp.ndarray:
        # sample random parameters
        thetas, betas, ws = self.sample_basis(key, Ns, amp, ls)
        # fourier basis part
        samps = vmap(
            lambda ti: vmap(lambda thi, bi, wi: jnp.dot(wi, self.phi(ti, thi, bi)))(
                thetas, betas, ws
            )
        )(t)

        # canonical basis part
        if vs is not None:
            _, LKvv = self.compute_covariances(amp, ls)
            qs = vmap(lambda vi, thi, bi, wi: self.compute_q(vi, LKvv, thi, bi, wi))(
                vs, thetas, betas, ws
            )  # Ns x Nz
            kv = map2matrix(self.kernel, t, self.z, amp, ls)  # Nt x Nz
            kv = jnp.einsum("ij, kj", qs, kv)  # Nt x Ns

            samps += kv.T

        return samps

    def sample(
        self, t: jnp.ndarray, Ns: int = 100, key: jrnd.PRNGKey = jrnd.PRNGKey(1)
    ) -> jnp.ndarray:
        try:
            assert t.shape[1] == self.D
        except AssertionError:
            raise ValueError("Dimension of input does not match dimension of GP.")

        if self.v is not None and len(self.v.shape) == 1:
            vs = jnp.tile(self.v.T, (Ns, 1))
        else:
            vs = self.v

        return self._sample(t, vs, self.amp, self.ls, Ns, key)


class SepEQApproxGP(EQApproxGP):
    """
    Class for high dim (>2) that are separable, ie have a different GP for each dimension.
    We further assume that the lengthscale and amp for each dimension is the same as well as the
    inducing inputs, so z is Nx1 v is NxD
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @partial(jit, static_argnums=(0, 2))
    def sample_basis(
        self, key: jrnd.PRNGKey, Ns: int, amp: float, ls: float
    ) -> Tuple[jnp.ndarray]:
        skey = jrnd.split(key, 3)
        thetas = self.sample_thetas(skey[0], (Ns, self.D, self.N_basis, 1), ls)
        betas = self.sample_betas(skey[1], (Ns, self.D, self.N_basis))
        ws = self.sample_ws(skey[2], (Ns, self.D, self.N_basis), amp)
        return thetas, betas, ws

    @partial(jit, static_argnums=(0, 5))
    def _sample(
        self,
        t: jnp.ndarray,
        vs: jnp.ndarray,
        amp: float,
        ls: float,
        Ns: int,
        key: jrnd.PRNGKey,
    ) -> jnp.ndarray:
        # sample random parameters
        thetas, betas, ws = self.sample_basis(key, Ns, amp, ls)
        # fourier basis part
        samps = vmap(
            lambda ti: vmap(
                lambda thi, bi, wi: vmap(
                    lambda tij, thij, bij, wij: jnp.dot(wij, self.phi(tij, thij, bij))
                )(ti, thi, bi, wi)
            )(thetas, betas, ws)
        )(jnp.expand_dims(t, 2)).swapaxes(
            0, 1
        )  # Nt x Ns x Nd

        # canonical basis part
        if vs is not None:

            _, LKvv = self.compute_covariances(amp, ls)
            qs = vmap(
                lambda vi, thi, bi, wi: vmap(
                    lambda vij, thij, bij, wij: self.compute_q(
                        vij, LKvv, thij, bij, wij
                    )
                )(vi, thi, bi, wi)
            )(
                vs, thetas, betas, ws
            )  # Ns x Nz x Nd
            kv = map2matrix(
                lambda ti, zj: vmap(lambda tij: self.kernel(tij, zj, amp, ls))(ti),
                t,
                self.z,
            )  # Nt x Nd x Nz

            kv = jnp.einsum("ijk, nkj->inj", qs, kv)  # Nt x Ns x Nd
            samps += kv

        return jnp.prod(samps, axis=2).T


class VarEQApproxGP(EQApproxGP):
    def __init__(
        self,
        z,
        mu=None,
        LC=None,
        N_basis: int = 500,
        D: int = 1,
        ls: float = 1.0,
        amp: float = 1.0,
        noise: float = 0.0,
        q_frac: float = 0.8,
        key=jrnd.PRNGKey(0),
    ):
        self.z = z
        self.N_basis = N_basis
        self.D = D

        self.ls = ls
        self.pr = l2p(ls)
        self.amp = amp
        self.noise = noise

        self.Kvv, self.LKvv = self.compute_covariances(amp, ls)

        if mu is None and LC is None:
            self.mu = super()._sample(self.z, None, self.amp, self.ls, 1, key).flatten()
            self.LC = self.LKvv * q_frac
        else:
            self.mu = mu
            self.LC = LC

    @staticmethod
    @jit
    def _KL(LC, mu, LK):
        C = LC @ LC.T
        mt = -0.5 * (
            jnp.dot(mu.T, jsp.linalg.cho_solve((LK, True), mu))
            + jnp.trace(jsp.linalg.cho_solve((LK, True), C))
        )

        st = 0.5 * (jnp.sum(jnp.log(jnp.diag(LC))) - jnp.sum(jnp.log(jnp.diag(LK))))
        return mt + st + 0.5 * LC.shape[0]

    # def _sample_vs
    def KL(self):
        return self._KL(self.LC, self.mu, self.LKvv)

    # @staticmethod
    @partial(
        jit,
        static_argnums=(
            0,
            3,
        ),
    )
    def _sample_vs(self, LC, mu, N_s, key):
        return jrnd.multivariate_normal(
            key, mu, JITTER * jnp.eye(mu.shape[0]) + LC @ LC.T, (N_s,)
        )

    def sample_vs(self, N_s, key):
        return self._sample_vs(self.LC, self.mu, N_s, key)

    @partial(jit, static_argnums=(0, 6))
    def _sample(
        self,
        t: jnp.ndarray,
        mu: jnp.ndarray,
        LC: jnp.ndarray,
        amp: float,
        ls: float,
        Ns: int,
        key: jrnd.PRNGKey,
    ) -> jnp.ndarray:
        keys = jrnd.split(key, 2)
        vs = self._sample_vs(LC, mu, Ns, keys[0])
        return super()._sample(t, vs, amp, ls, Ns, keys[1])

    def sample(self, t, Ns, key):
        return self._sample(t, self.mu, self.LC, self.amp, self.ls, Ns, key)

    @partial(jit, static_argnums=(0, 6))
    def sample_qs(self, mu, LC, amp, ls, basis_samps, Ns, key):
        vs = self._sample_vs(LC, mu, Ns, key)
        _, LK = self.compute_covariances(amp, ls)
        return vmap(lambda vui, thi, bi, wi: self.compute_q(vui, LK, thi, bi, wi))(
            vs, *basis_samps
        )


class SepVarEQApproxGP(SepEQApproxGP, VarEQApproxGP):
    def __init__(
        self,
        z,
        mu=None,
        LC=None,
        N_basis: int = 500,
        D: int = 1,
        ls: float = 1.0,
        amp: float = 1.0,
        noise: float = 0.0,
        q_frac: float = 0.8,
        key=jrnd.PRNGKey(0),
    ):
        self.z = z
        self.N_basis = N_basis
        self.D = D

        self.ls = ls
        self.pr = l2p(ls)
        self.amp = amp
        self.noise = noise

        self.Kvv, self.LKvv = self.compute_covariances(amp, ls)

        if mu is None and LC is None:
            # FIX
            self.mu = jnp.zeros((z.shape[0], D))
            self.LC = jnp.repeat(jnp.expand_dims(self.LKvv * q_frac, 2), self.D, axis=2)
        else:
            self.mu = mu
            self.LC = LC

    @staticmethod
    @jit
    def _KL(LC, mu, LK):
        return jnp.sum(vmap(VarEQApproxGP._KL, in_axes=(2, 1, None))(LC, mu, LK))

    @partial(jit, static_argnums=(0, 6))
    def sample_qs(self, mu, LC, amp, ls, basis_samps, Ns, key):
        vs = vmap(self._sample_vs, in_axes=(2, 1, None, 0))(
            LC, mu, Ns, jrnd.split(key, self.D)
        ).swapaxes(0, 1)
        _, LK = self.compute_covariances(amp, ls)
        return vmap(
            lambda vi, thi, bi, wi: vmap(
                lambda vij, thij, bij, wij: self.compute_q(vij, LK, thij, bij, wij)
            )(vi, thi, bi, wi)
        )(vs, *basis_samps)

    @partial(jit, static_argnums=(0, 6))
    def _sample(
        self,
        t: jnp.ndarray,
        mu: jnp.ndarray,
        LC: jnp.ndarray,
        amp: float,
        ls: float,
        Ns: int,
        key: jrnd.PRNGKey,
    ) -> jnp.ndarray:
        keys = jrnd.split(key, 2)
        vsl = vmap(self._sample_vs, in_axes=(2, 1, None, 0))(
            LC, mu, Ns, jrnd.split(keys[0], self.D)
        ).swapaxes(0, 1)
        return super()._sample(t, vsl, amp, ls, Ns, keys[1])

    def sample(self, t, Ns, key):
        return self._sample(t, self.mu, self.LC, self.amp, self.ls, Ns, key)


class MOVarNVKM:
    def __init__(
        self,
        zgs: List[List[jnp.ndarray]],
        zu: jnp.ndarray,
        data: Tuple[List[jnp.ndarray]],
        q_frac: float = 0.8,
        key: jrnd.PRNGKey = None,
        likelihood: Callable = gaussian_likelihood,
        N_basis: int = 500,
        ampgs: List[List[float]] = [[1.0], [1.0]],
        noise: List[float] = [1.0, 1.0],
        alpha: List[List[float]] = [[1.0], [1.0]],
        lsgs: List[List[float]] = [[1.0], [1.0]],
        lsu: float = 1.0,
        ampu: float = 1.0,
    ):
        """
        Implements the NVKM, for multiple ouputs.

        Args:
            zgs (List[List[jnp.ndarray]]): Inducing inputs for each Volterra kernel,
            each ouput is first dimension, each term for that output second dimesion.
            zu (jnp.ndarray): Inducing inputs for the input process u.
            data (Tuple[List[jnp.ndarray]]): data for each ouput, in format (x, y).
            q_class (type, optional): Varational distribution. Defaults to MOIndependentGaussians.
            q_pars_init (Dict[str, List[List[jnp.ndarray]]], optional): Initial vairational parameters.
            Defaults to None.
            q_initializer_pars (float, optional): If variational parmeters not given,
             then initialser will be used, which aslo takes parameters, given here. Defaults to None.
            q_init_key (jrnd.PRNGKey, optional): random key for initialiser. Defaults to None.
            likelihood (Callable, optional): Likelihood function for data. Defaults to gaussian_likelihood.
            N_basis (int, optional): Number of basis functions. Defaults to 500.
            ampgs (List[List[float]], optional): amplitudes of each Volterra kernel. Defaults to [[1.0], [1.0]].
            noise (List[float], optional): Noise for each output Defaults to [1.0, 1.0].
            alpha (List[List[float]], optional): Decay for each Volterra kernel. Defaults to [[1.0], [1.0]].
            lsgs (List[List[float]], optional): Length scale for each Volterra kernel. Defaults to [[1.0], [1.0]].
            lsu (float, optional): [description]. Length scale for input process. to 1.0.
            ampu (float, optional): [description]. Amplitude for input process. to 1.0.
        """

        self.zgs = zgs
        self.vgs = None
        self.zu = zu
        self.vu = None

        self.N_basis = N_basis
        self.C = [len(l) for l in zgs]
        self.O = len(zgs)
        self.noise = noise
        self.alpha = alpha

        self.lsgs = lsgs
        self.lsu = lsu
        self.ampgs = ampgs
        self.ampu = ampu

        self.u_gp, self.g_gps = self.construct_gps(lsu, lsgs, q_frac=q_frac, key=key)

        self.data = data
        self.likelihood = likelihood
        self.I_class = Full
        # self.q_of_v = q_class()
        # if q_pars_init is None:
        #     q_pars_init = self.q_of_v.initialize(
        #         self, q_initializer_pars, key=q_init_key
        #     )
        self.q_pars = self.construct_q_pars()

    def construct_gps(self, lsu, lsgs, key=jrnd.PRNGKey(0), q_frac=0.8, q_pars=None):
        g_gps = []
        for i in range(self.O):
            gpl = []
            for j in range(self.C[i]):

                if q_pars is not None:
                    gpl.append(
                        VarEQApproxGP(
                            z=self.zgs[i][j],
                            N_basis=self.N_basis,
                            mu=q_pars["mu_gs"][i][j],
                            LC=q_pars["LC_gs"][i][j],
                            D=j + 1,
                            ls=lsgs[i][j],
                            amp=1.0,
                        )
                    )
                else:

                    gpl.append(
                        VarEQApproxGP(
                            z=self.zgs[i][j],
                            N_basis=self.N_basis,
                            q_frac=q_frac,
                            D=j + 1,
                            ls=lsgs[i][j],
                            amp=1.0,
                            key=key,
                        )
                    )

                key, _ = jrnd.split(key, 2)
            g_gps.append(gpl)

        u_gp = VarEQApproxGP(
            z=self.zu, N_basis=self.N_basis, D=1, ls=lsu, amp=1.0, key=key
        )
        return u_gp, g_gps

    def construct_q_pars(self):
        return {
            "mu_u": self.u_gp.mu,
            "LC_us": self.u_gp.LC,
            "mu_gs": [[gp.mu for gp in gpl] for gpl in self.g_gps],
            "LC_gs": [[gp.LC for gp in gpl] for gpl in self.g_gps],
        }

    # # def set_u_gp(self, ampu, lsu):
    #     return EQApproxGP(
    #         z=self.zu, v=None, N_basis=self.N_basis, D=1, ls=lsu, amp=ampu
    #     )

    @partial(jit, static_argnums=(0,))
    def _KL(self, mu_u, mu_gs, LC_us, LC_gs, lsu, lsgs):
        out = 0.0
        for i in range(len(self.g_gps)):
            for j in range(len(self.g_gps[i])):
                _, LK = self.g_gps[i][j].compute_covariances(1.0, lsgs[i][j])
                out += self.g_gps[i][j]._KL(LC_gs[i][j], mu_gs[i][j], LK)
        _, LKu = self.u_gp.compute_covariances(1.0, lsu)
        out += self.u_gp._KL(LC_us, mu_u, LKu)
        return out

    # @partial(jit, static_argnums=(0,))
    # def _compute_p_pars(self, ampgs, lsgs, ampu, lsu):
    #     return {
    #         "LK_gs": [
    #             [
    #                 self.g_gps[i][j].compute_covariances(1.0, lsgs[i][j])[1]
    #                 for j in range(self.C[i])
    #             ]
    #             for i in range(self.O)
    #         ],
    #         "LK_u": self.u_gp.compute_covariances(ampu, lsu)[1],
    #     }

    # def sample_diag_g_gps(self, ts, N_s, keys):
    #     vs = self.q_of_v.sample(self.q_pars, N_s, keys[0])["gs"]
    #     samps = []
    #     for i in range(self.O):
    #         il = []
    #         for j, gp in enumerate(self.g_gps[i]):
    #             keys = jrnd.split(keys[1])
    #             il.append(
    #                 gp._sample(ts[i][j], vs[i][j], 1.0, self.lsgs[i][j], N_s, keys[1],)
    #             )
    #         samps.append(il)

    #     return samps

    # def sample_u_gp(self, t, N_s, keys):
    #     vs = self.q_of_v.sample(self.q_pars, N_s, keys[0])["u"]
    #     return self.u_gp._sample(
    #         t.reshape(-1, 1), vs, self.ampu, self.lsu, N_s, keys[1]
    #     )

    @partial(jit, static_argnums=(0, 7))
    def _sample(self, ts, q_pars, ampgs, lsgs, ampu, lsu, N_s, key):

        keys = jrnd.split(key, 2)
        u_basis_samps = self.u_gp.sample_basis(keys[1], N_s, ampu, lsu)
        thetaul, betaul, wul = u_basis_samps
        # _, u_LKvv = u_gp.compute_covariances(ampu, lsu)

        # qul = vmap(lambda vui, thi, bi, wi: u_gp.compute_q(vui, u_LKvv, thi, bi, wi))(
        #     v_samps["u"], thetaul, betaul, wul
        # )
        qul = self.u_gp.sample_qs(
            q_pars["mu_u"], q_pars["LC_us"], ampu, lsu, u_basis_samps, N_s, keys[0]
        )
        samps = []
        for i in range(self.O):
            if ts[i] is None:
                samps.append(None)
                continue
            sampsi = jnp.zeros((len(ts[i]), N_s))
            for j in range(0, self.C[i]):
                keys = jrnd.split(keys[-1], 3)
                g_gp_i = self.g_gps[i][j]

                g_basis_samps = g_gp_i.sample_basis(keys[0], N_s, 1.0, lsgs[i][j])
                thetagl, betagl, wgl = g_basis_samps
                qgl = g_gp_i.sample_qs(
                    q_pars["mu_gs"][i][j],
                    q_pars["LC_gs"][i][j],
                    1.0,
                    lsgs[i][j],
                    g_basis_samps,
                    N_s,
                    keys[1],
                )

                sampsi += (
                    ampu
                    * ampgs[i][j] ** (j + 1)
                    * self.I_class.I(
                        ts[i],
                        g_gp_i.z,
                        self.u_gp.z,
                        thetagl,
                        betagl,
                        thetaul,
                        betaul,
                        wgl,
                        qgl,
                        wul,
                        qul,
                        self.alpha[i][j],
                        l2p(lsgs[i][j]),
                        l2p(lsu),
                    )
                )
                # id_print(sampsi)
            samps.append(sampsi)
        return samps

    def sample(self, ts, N_s, key=jrnd.PRNGKey(1)):
        return self._sample(
            ts,
            self.q_pars,
            self.ampgs,
            self.lsgs,
            self.ampu,
            self.lsu,
            N_s,
            key,
        )

    def predict(self, ts, N_s, key=jrnd.PRNGKey(1)):
        samps = self.sample(ts, N_s, key=key)
        return (
            [jnp.mean(si, axis=1) if si is not None else si for si in samps],
            [
                jnp.var(si, axis=1) + self.noise[i] ** 2 if si is not None else si
                for i, si in enumerate(samps)
            ],
        )

    @partial(jit, static_argnums=(0, 8))
    def _compute_bound(self, data, q_pars, ampgs, lsgs, ampu, lsu, noise, N_s, key):
        # p_pars = self._compute_p_pars(ampgs, lsgs, ampu, lsu)

        # for i in range(self.O):
        #     for j in range(self.C[i]):
        #         q_pars["LC_gs"][i][j] = choleskyize(q_pars["LC_gs"][i][j])
        # q_pars["LC_u"] = choleskyize(q_pars["LC_u"])

        KL = self._KL(
            q_pars["mu_u"], q_pars["mu_gs"], q_pars["LC_us"], q_pars["LC_gs"], lsu, lsgs
        )

        xs, ys = data
        samples = self._sample(
            xs,
            q_pars,
            ampgs,
            lsgs,
            ampu,
            lsu,
            N_s,
            key,
        )
        like = 0.0
        for i in range(self.O):
            like += self.likelihood(ys[i], samples[i], noise[i])
        return -(KL + like)

    def compute_bound(self, N_s, key=jrnd.PRNGKey(1)):
        return self._compute_bound(
            self.data,
            self.q_pars,
            self.ampgs,
            self.lsgs,
            self.ampu,
            self.lsu,
            self.noise,
            N_s,
            key,
        )

    def fit(self, its, lr, batch_size, N_s, dont_fit=[], key=jrnd.PRNGKey(1)):

        xs, ys = self.data

        std_fit = ["q_pars", "ampgs", "lsgs", "ampu", "lsu", "noise"]
        std_argnums = list(range(1, 7))
        dpars_init = []
        dpars_argnum = []
        bound_arg = [0.0] * 6
        for i, k in enumerate(std_fit):
            if k not in dont_fit:
                dpars_init.append(getattr(self, k))
                dpars_argnum.append(std_argnums[i])
            bound_arg[i] = getattr(self, k)

        grad_fn = jit(
            value_and_grad(self._compute_bound, argnums=dpars_argnum),
            static_argnums=(7,),
        )

        opt_init, opt_update, get_params = opt.adam(lr)

        opt_state = opt_init(tuple(dpars_init))

        for i in range(its):
            skey, key = jrnd.split(key, 2)
            y_bs = []
            x_bs = []
            for j in range(self.O):
                skey, key = jrnd.split(key, 2)

                if batch_size:
                    rnd_idx = jrnd.choice(key, len(ys[j]), shape=(batch_size,))
                    y_bs.append(ys[j][rnd_idx])
                    x_bs.append(xs[j][rnd_idx])
                else:
                    y_bs.append(ys[j])
                    x_bs.append(xs[j])

            for k, ix in enumerate(dpars_argnum):
                bound_arg[ix - 1] = get_params(opt_state)[k]
            value, grads = grad_fn(
                (x_bs, y_bs),
                *bound_arg,
                N_s,
                skey,
            )

            if jnp.any(jnp.isnan(value)):
                print("nan F!!")
                return get_params(opt_state)

            if i % 10 == 0:
                print(f"it: {i} F: {value} ")

            opt_state = opt_update(i, grads, opt_state)

        print(bound_arg[1])
        for i, ix in enumerate(dpars_argnum):
            bound_arg[ix - 1] = get_params(opt_state)[i]

        for i, k in enumerate(std_fit):
            setattr(self, k, bound_arg[i])

        # self.opt_triple = opt_init, opt_update, get_params
        self.u_gp, self.g_gps = self.construct_gps(
            self.lsu, self.lsgs, q_pars=self.q_pars
        )

    def save(self, f_name):
        sd = {}
        for k, v in self.__dict__.items():
            if k not in ["q_of_v", "likelihood", "g_gps", "u_gps", "p_pars"]:
                sd[k] = v

        with open(f_name, "wb") as file:
            pickle.dump(sd, file)

    def plot_samples(
        self, tu, tys, N_s, return_axs=False, save=False, key=jrnd.PRNGKey(304)
    ):

        fig, axs = plt.subplots(self.O + 1, 1, figsize=(10, 3.5 * (1 + self.O)))

        keys = jrnd.split(key, 2)
        u_samps = self.u_gp.sample(tu.reshape(-1, 1), N_s, keys[0])
        axs[0].set_ylabel(f"$u$")
        axs[0].set_xlabel("$t$")
        axs[0].scatter(self.zu, self.q_pars["mu_u"], c="green", alpha=0.5, s=5.0)
        axs[0].plot(tu, u_samps, c="blue", alpha=0.5)

        samps = self.sample(tys, N_s, keys[1])
        for i in range(0, self.O):
            axs[i + 1].set_ylabel(f"$y_{i+1}$")
            axs[i + 1].set_xlabel("$t$")
            axs[i + 1].plot(tys[i], samps[i], c="green", alpha=0.5)
            axs[i + 1].scatter(
                self.data[0][i], self.data[1][i], c="blue", alpha=0.5, s=5.0
            )

        if return_axs:
            return axs

        if save:
            plt.savefig(save)
        plt.show()

    def plot_filters(
        self, tf, N_s, return_axs=False, save=False, key=jrnd.PRNGKey(211)
    ):
        tfs = [
            [jnp.vstack((tf for j in range(gp.D))).T for gp in self.g_gps[i]]
            for i in range(self.O)
        ]
        _, axs = plt.subplots(
            ncols=max(self.C),
            nrows=self.O,
            figsize=(4 * max(self.C), 2 * self.O),
        )
        if max(self.C) == 1 and self.O == 1:
            y = self.g_gps[0][0].sample(tfs[0][0], N_s, key).T * jnp.exp(
                -self.alpha[0][0] * (tf) ** 2
            )
            axs.plot(tf, y.T, c="red", alpha=0.5)
            axs.set_title("$G_{%s, %s}$" % (1, 1))

        elif max(self.C) == 1:
            for i in range(self.O):
                y = self.g_gps[i][0].sample(tfs[i][0], N_s, key).T * jnp.exp(
                    -self.alpha[i][0] * (tf) ** 2
                )
                axs[i].plot(tf, y.T, c="red", alpha=0.5)
                axs[i].set_title("$G_{%s, %s}$" % (i + 1, 1))
                key, _ = jrnd.split(key, 2)

        elif self.O == 1:
            for j in range(self.C[0]):
                y = self.g_gps[0][j].sample(tfs[0][j], N_s, key).T * jnp.exp(
                    -self.alpha[0][j] * (tf) ** 2
                )
                axs[j].plot(tf, y.T, c="red", alpha=0.5)
                axs[j].set_title("$G_{%s, %s}$" % (1, j + 1))
                key, _ = jrnd.split(key, 2)

        else:
            for i in range(self.O):
                for j in range(self.C[i]):
                    y = self.g_gps[i][j].sample(tfs[i][j], N_s, key).T * jnp.exp(
                        -self.alpha[i][j] * (tf) ** 2
                    )
                    axs[i][j].plot(tf, y.T, c="red", alpha=0.5)
                    axs[i][j].set_title("$G_{%s, %s}$" % (i + 1, j + 1))
                    key, _ = jrnd.split(key, 2)
                for k in range(self.C[i], max(self.C)):
                    axs[i][k].axis("off")

        plt.tight_layout()

        if return_axs:
            return axs

        if save:
            plt.savefig(save)
        plt.show()


class SepMOVarNVKM(MOVarNVKM):
    def __init__(self, *args, causal=False, **kwargs):
        super().__init__(*args, **kwargs)
        if causal:
            self.I_class = CausalSeparable
        else:
            self.I_class = Separable

    def construct_gps(self, lsu, lsgs, key=jrnd.PRNGKey(0), q_frac=0.8, q_pars=None):
        g_gps = []
        for i in range(self.O):
            gpl = []
            for j in range(self.C[i]):
                key, _ = jrnd.split(key)
                if q_pars is not None:
                    gpl.append(
                        SepVarEQApproxGP(
                            z=self.zgs[i][j],
                            N_basis=self.N_basis,
                            mu=q_pars["mu_gs"][i][j],
                            LC=q_pars["LC_gs"][i][j],
                            D=j + 1,
                            ls=lsgs[i][j],
                            amp=1.0,
                        )
                    )
                else:
                    gpl.append(
                        SepVarEQApproxGP(
                            z=self.zgs[i][j],
                            N_basis=self.N_basis,
                            q_frac=q_frac,
                            D=j + 1,
                            ls=lsgs[i][j],
                            amp=1.0,
                            key=key,
                        )
                    )

                key, _ = jrnd.split(key)
            g_gps.append(gpl)

        if q_pars is not None:
            u_gp = VarEQApproxGP(
                z=self.zu,
                N_basis=self.N_basis,
                mu=q_pars["mu_u"],
                LC=q_pars["LC_us"],
                D=1,
                ls=lsu,
                amp=1.0,
                key=key,
            )
        else:
            u_gp = VarEQApproxGP(
                z=self.zu,
                N_basis=self.N_basis,
                q_frac=q_frac,
                D=1,
                ls=lsu,
                amp=1.0,
                key=key,
            )
        return u_gp, g_gps


class SepHomogMOVarNVKM(MOVarNVKM):
    def __init__(self, *args, causal=False, **kwargs):
        super().__init__(*args, **kwargs)

        if causal:
            self.I_class = CausalHomogeneous
        else:
            self.I_class = Homogeneous

    def construct_gps(self, lsu, lsgs, key=jrnd.PRNGKey(0), q_frac=0.8, q_pars=None):
        g_gps = []
        for i in range(self.O):
            gpl = []
            for j in range(self.C[i]):
                key, _ = jrnd.split(key)
                if q_pars is not None:
                    gpl.append(
                        VarEQApproxGP(
                            z=self.zgs[i][j],
                            N_basis=self.N_basis,
                            mu=q_pars["mu_gs"][i][j],
                            LC=q_pars["LC_gs"][i][j],
                            D=1,
                            ls=lsgs[i][j],
                            amp=1.0,
                        )
                    )
                else:
                    gpl.append(
                        VarEQApproxGP(
                            z=self.zgs[i][j],
                            N_basis=self.N_basis,
                            q_frac=q_frac,
                            D=1,
                            ls=lsgs[i][j],
                            amp=1.0,
                            key=key,
                        )
                    )

                key, _ = jrnd.split(key)
            g_gps.append(gpl)

        if q_pars is not None:
            u_gp = VarEQApproxGP(
                z=self.zu,
                N_basis=self.N_basis,
                mu=q_pars["mu_u"],
                LC=q_pars["LC_us"],
                D=1,
                ls=lsu,
                amp=1.0,
                key=key,
            )
        else:
            u_gp = VarEQApproxGP(
                z=self.zu,
                N_basis=self.N_basis,
                q_frac=q_frac,
                D=1,
                ls=lsu,
                amp=1.0,
                key=key,
            )
        return u_gp, g_gps

    @partial(jit, static_argnums=(0, 7))
    def _sample(self, ts, q_pars, ampgs, lsgs, ampu, lsu, N_s, key):

        keys = jrnd.split(key, 2)
        u_basis_samps = self.u_gp.sample_basis(keys[1], N_s, ampu, lsu)
        thetaul, betaul, wul = u_basis_samps
        # _, u_LKvv = u_gp.compute_covariances(ampu, lsu)

        # qul = vmap(lambda vui, thi, bi, wi: u_gp.compute_q(vui, u_LKvv, thi, bi, wi))(
        #     v_samps["u"], thetaul, betaul, wul
        # )
        qul = self.u_gp.sample_qs(
            q_pars["mu_u"], q_pars["LC_us"], ampu, lsu, u_basis_samps, N_s, keys[0]
        )
        samps = []
        for i in range(self.O):
            if ts[i] is None:
                samps.append(None)
                continue
            sampsi = jnp.zeros((len(ts[i]), N_s))
            for j in range(0, self.C[i]):
                keys = jrnd.split(keys[-1], 3)
                g_gp_i = self.g_gps[i][j]

                g_basis_samps = g_gp_i.sample_basis(keys[0], N_s, 1.0, lsgs[i][j])
                thetagl, betagl, wgl = g_basis_samps
                qgl = g_gp_i.sample_qs(
                    q_pars["mu_gs"][i][j],
                    q_pars["LC_gs"][i][j],
                    1.0,
                    lsgs[i][j],
                    g_basis_samps,
                    N_s,
                    keys[1],
                )

                sampsi += (
                    ampu
                    * ampgs[i][j] ** (j + 1)
                    * self.I_class.I(
                        ts[i],
                        g_gp_i.z,
                        self.u_gp.z,
                        thetagl[:, :, 0],
                        betagl,
                        thetaul,
                        betaul,
                        wgl,
                        qgl,
                        wul,
                        qul,
                        self.alpha[i][j],
                        l2p(lsgs[i][j]),
                        l2p(lsu),
                    )
                    ** (j + 1)
                )

            samps.append(sampsi)

        return samps

    def plot_filters(
        self, tf, N_s, return_axs=False, save=False, key=jrnd.PRNGKey(211)
    ):
        tfs = [[tf.reshape(-1, 1) for gp in self.g_gps[i]] for i in range(self.O)]
        _, axs = plt.subplots(
            ncols=max(self.C),
            nrows=self.O,
            figsize=(4 * max(self.C), 2 * self.O),
        )
        if max(self.C) == 1 and self.O == 1:
            y = self.g_gps[0][0].sample(tfs[0][0], N_s, key).T * jnp.exp(
                -self.alpha[0][0] * (tf) ** 2
            )
            axs.plot(tf, y.T, c="red", alpha=0.5)
            axs.set_title("$G_{%s, %s}$" % (1, 1))

        elif max(self.C) == 1:
            for i in range(self.O):
                y = self.g_gps[i][0].sample(tfs[i][0], N_s, key).T * jnp.exp(
                    -self.alpha[i][0] * (tf) ** 2
                )
                axs[i].plot(tf, y.T, c="red", alpha=0.5)
                axs[i].set_title("$G_{%s, %s}$" % (i + 1, 1))
                key, _ = jrnd.split(key, 2)

        elif self.O == 1:
            for j in range(self.C[0]):
                y = self.g_gps[0][j].sample(tfs[0][j], N_s, key).T * jnp.exp(
                    -self.alpha[0][j] * (tf) ** 2
                )
                axs[j].plot(tf, y.T, c="red", alpha=0.5)
                axs[j].set_title("$G_{%s, %s}$" % (1, j + 1))
                key, _ = jrnd.split(key, 2)

        else:
            for i in range(self.O):
                for j in range(self.C[i]):
                    y = self.g_gps[i][j].sample(tfs[i][j], N_s, key).T * jnp.exp(
                        -self.alpha[i][j] * (tf) ** 2
                    )
                    axs[i][j].plot(tf, y.T, c="red", alpha=0.5)
                    axs[i][j].set_title("$G_{%s, %s}$" % (i + 1, j + 1))
                    key, _ = jrnd.split(key, 2)
                for k in range(self.C[i], max(self.C)):
                    axs[i][k].axis("off")

        plt.tight_layout()

        if return_axs:
            return axs

        if save:
            plt.savefig(save)
        plt.show()


class IOMOVarNVKM(MOVarNVKM):
    def __init__(
        self,
        zgs: List[List[jnp.ndarray]],
        zu: jnp.ndarray,
        u_data: Tuple[jnp.ndarray],
        y_data: Tuple[List[jnp.ndarray]],
        u_noise: float = 1.0,
        **kwargs,
    ):
        """
        Implements the input output variant of the NVKM.

        Args:
            zgs (List[List[jnp.ndarray]]): Inducing inputs for each Volterra kernel,
            each ouput is first dimension, each term for that output second dimesion.
            zu (jnp.ndarray): Inducing inputs for the input process u.
            u_data (Tuple[jnp.ndarray]): data for the input process in form (x, y)
            y_data (Tuple[List[jnp.ndarray]]): data for the each ouput in form (x, y)
            u_noise (float, optional): noise for input process. Defaults to 1.0.
        """
        super().__init__(
            zgs,
            zu,
            None,
            **kwargs,
        )
        self.u_noise = u_noise
        self.data = (u_data, y_data)

    @partial(jit, static_argnums=(0, 8))
    def _joint_sample(self, tu, tys, q_pars, ampgs, lsgs, ampu, lsu, N_s, key):
        # we need to use the same key to make sure the samples are consistent
        u_samps = self.u_gp._sample(
            tu.reshape(-1, 1), q_pars["mu_u"], q_pars["LC_us"], ampu, lsu, N_s, key
        )
        y_samps = self._sample(tys, q_pars, ampgs, lsgs, ampu, lsu, N_s, key)
        return u_samps, y_samps

    def joint_sample(self, tu, tys, N_s, key=jrnd.PRNGKey(1)):
        return self._joint_sample(
            tu,
            tys,
            self.q_pars,
            self.ampgs,
            self.lsgs,
            self.ampu,
            self.lsu,
            N_s,
            key,
        )

    @partial(jit, static_argnums=(0, 9))
    def _compute_bound(
        self, data, q_pars, ampgs, lsgs, ampu, lsu, noise, u_noise, N_s, key
    ):

        # p_pars = self._compute_p_pars(ampgs, lsgs, ampu, lsu)
        # for i in range(self.O):
        #     for j in range(self.C[i]):
        #         q_pars["LC_gs"][i][j] = choleskyize(q_pars["LC_gs"][i][j])
        # q_pars["LC_u"] = choleskyize(q_pars["LC_u"])

        KL = self._KL(
            q_pars["mu_u"], q_pars["mu_gs"], q_pars["LC_us"], q_pars["LC_gs"], lsu, lsgs
        )

        u_data, y_data = data
        xs, ys = y_data
        ui, uo = u_data

        u_samples, y_samples = self._joint_sample(
            ui, xs, q_pars, ampgs, lsgs, ampu, lsu, N_s, key
        )
        like = 0.0
        for i in range(self.O):
            like += self.likelihood(ys[i], y_samples[i], noise[i])
        like += self.likelihood(uo, u_samples, u_noise)
        return -(KL + like)

    def fit(
        self,
        its,
        lr,
        batch_size,
        N_s,
        dont_fit=[],
        key=jrnd.PRNGKey(1),
    ):

        u_data, y_data = self.data
        xu, yu = u_data
        xs, ys = y_data

        std_fit = ["q_pars", "ampgs", "lsgs", "ampu", "lsu", "noise", "u_noise"]
        std_argnums = list(range(1, 8))
        opt_init, opt_update, get_params = opt.adam(lr)
        dpars_init = []
        dpars_argnum = []
        bound_arg = [0.0] * 7
        for i, k in enumerate(std_fit):
            if k not in dont_fit:
                dpars_init.append(getattr(self, k))
                dpars_argnum.append(std_argnums[i])
            bound_arg[i] = getattr(self, k)

        grad_fn = jit(
            value_and_grad(self._compute_bound, argnums=dpars_argnum),
            static_argnums=(8,),
        )
        opt_state = opt_init(tuple(dpars_init))

        for i in range(its):
            skey, key = jrnd.split(key, 2)

            if batch_size:
                bkeys = jrnd.split(key, self.O + 1)
                u_rnd_idx = jrnd.choice(bkeys[0], len(yu), shape=(batch_size,))
                yu_bs = yu[u_rnd_idx]
                xu_bs = xu[u_rnd_idx]

                y_bs = []
                x_bs = []
                for j in range(self.O):
                    y_rnd_idx = jrnd.choice(
                        bkeys[j + 1], len(ys[j]), shape=(batch_size,)
                    )
                    y_bs.append(ys[j][y_rnd_idx])
                    x_bs.append(xs[j][y_rnd_idx])
            else:
                xu_bs = xu
                yu_bs = yu

                x_bs = xs
                y_bs = ys

            for k, ix in enumerate(dpars_argnum):
                bound_arg[ix - 1] = get_params(opt_state)[k]
            value, grads = grad_fn(
                ((xu_bs, yu_bs), (x_bs, y_bs)),
                *bound_arg,
                N_s,
                skey,
            )
            if jnp.any(jnp.isnan(value)):
                print("nan F!!")
                return get_params(opt_state)

            if i % 10 == 0:
                print(f"it: {i} F: {value} ")

            opt_state = opt_update(i, grads, opt_state)

        for i, ix in enumerate(dpars_argnum):
            bound_arg[ix - 1] = get_params(opt_state)[i]

        for i, k in enumerate(std_fit):
            setattr(self, k, bound_arg[i])

        # self.opt_triple = opt_init, opt_update, get_params
        self.u_gp, self.g_gps = self.construct_gps(
            self.lsu, self.lsgs, q_pars=self.q_pars
        )

    def plot_samples(
        self, tu, tys, N_s, return_axs=False, save=False, key=jrnd.PRNGKey(304)
    ):

        _, axs = plt.subplots(self.O + 1, 1, figsize=(15, 2.5 * (1 + self.O)))

        u_data, y_data = self.data
        u_samps, y_samps = self.joint_sample(tu, tys, N_s, key=key)
        axs[0].set_ylabel(f"$u$")
        axs[0].set_xlabel("$t$")
        axs[0].scatter(
            self.zu, self.q_pars["mu_u"], c="purple", marker="x", label="$\mu_u$"
        )
        axs[0].scatter(u_data[0], u_data[1], c="green", label="Data", alpha=0.5)
        axs[0].plot(tu, u_samps, c="blue", alpha=0.35)
        axs[0].legend()
        for i in range(0, self.O):
            axs[i + 1].set_ylabel(f"$y_{i+1}$")
            axs[i + 1].set_xlabel("$t$")
            axs[i + 1].plot(tys[i], y_samps[i], c="green", alpha=0.35)
            axs[i + 1].scatter(
                y_data[0][i], y_data[1][i], label="Data", c="blue", alpha=0.5
            )
            axs[i + 1].legend()

        if return_axs:
            return axs

        if save:
            plt.savefig(save)
        plt.show()


class SepHomogIOMOVarNVKM(IOMOVarNVKM, SepHomogMOVarNVKM):
    pass


def load_io_model(pkl_path: str) -> IOMOVarNVKM:
    """
    Loads IOVarNVKM from .pkl file.

    Args:
        pkl_path (str): path to .pkl file.

    Returns:
        IOMOVarNVKM: the model.
    """
    with open(pkl_path, "rb") as f:
        model_dict = pickle.load(f)

    return IOMOVarNVKM(
        model_dict["zgs"],
        model_dict["zu"],
        model_dict["data"][0],
        model_dict["data"][1],
        u_noise=model_dict["u_noise"],
        q_pars_init=model_dict["q_pars"],
        N_basis=model_dict["N_basis"],
        ampgs=model_dict["ampgs"],
        noise=model_dict["noise"],
        alpha=model_dict["alpha"],
        lsgs=model_dict["lsgs"],
        lsu=model_dict["lsu"],
        ampu=model_dict["ampu"],
    )


def load_mo_model(pkl_path: str) -> MOVarNVKM:
    """
    Loads MOVarNVKM from .pkl file.

    Args:
        pkl_path (str): path to .pkl file.

    Returns:
        MOVarNVKM: The model
    """
    with open(pkl_path, "rb") as f:
        model_dict = pickle.load(f)

    return MOVarNVKM(
        model_dict["zgs"],
        model_dict["zu"],
        model_dict["data"],
        q_pars_init=model_dict["q_pars"],
        N_basis=model_dict["N_basis"],
        ampgs=model_dict["ampgs"],
        noise=model_dict["noise"],
        alpha=model_dict["alpha"],
        lsgs=model_dict["lsgs"],
        lsu=model_dict["lsu"],
        ampu=model_dict["ampu"],
    )
