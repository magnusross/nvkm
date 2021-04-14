from functools import partial
from typing import Callable, List, Tuple, Union
import logging
from jax.config import config
import pickle
import jax.experimental.optimizers as opt
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy as jsp
from jax.ops import index_add, index
import matplotlib.pyplot as plt
from jax import jit, vmap, value_and_grad


from .integrals import fast_I, slow_I
from .settings import JITTER
from .utils import eq_kernel, l2p, map2matrix, vmap_scan
from .vi import (
    IndependentGaussians,
    VariationalDistribution,
    VIPars,
    gaussain_likelihood,
)

config.update("jax_enable_x64", True)


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

    # @partial(jit, static_argnums=(0,))
    def compute_covariances(self, amp, ls):
        Kvv = map2matrix(self.kernel, self.z, self.z, amp, ls) + (
            self.noise + JITTER
        ) * jnp.eye(self.z.shape[0])
        LKvv = jnp.linalg.cholesky(Kvv)
        return Kvv, LKvv

    # @partial(jit, static_argnums=(0,))
    # def fast_covariance_recompute(self, new_amp):
    #     factor = new_amp ** 2 / self.amp ** 2
    #     return factor * self.Kvv, factor * self.LKvv

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
    def sample_ws(self, key, shape, amp):
        return amp * jnp.sqrt(2 / self.N_basis) * jrnd.normal(key, shape)

    @partial(jit, static_argnums=(0,))
    def phi(self, t, theta, beta):
        return jnp.cos(jnp.dot(theta, t) + beta)

    @partial(jit, static_argnums=(0,))
    def compute_Phi(self, thetas, betas):
        return vmap(lambda zi: self.phi(zi, thetas, betas))(self.z)

    @partial(jit, static_argnums=(0,))
    def compute_q(self, v, LKvv, thetas, betas, ws):
        Phi = self.compute_Phi(thetas, betas)
        b = v - Phi @ ws
        return jsp.linalg.cho_solve((LKvv, True), b)

    @partial(jit, static_argnums=(0, 4))
    def _sample(self, t, v, amp, Ns, key=jrnd.PRNGKey(1)):

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
        ws = self.sample_ws(skey[2], (Ns, self.N_basis), amp)

        # fourier basis part
        samps = vmap(
            lambda ti: vmap(lambda thi, bi, wi: jnp.dot(wi, self.phi(ti, thi, bi)))(
                thetas, betas, ws
            )
        )(t)

        # canonical basis part
        if v is None:
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

        return self._sample(t, self.v, self.amp, Ns, key=key)


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
        wul = u_gp.sample_ws(skey[2], (N_s, u_gp.N_basis), u_gp.amp)

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
            wgl = G_gp_i.sample_ws(skey[2], (N_s, G_gp_i.N_basis), G_gp_i.amp)
            _, G_LKvv = G_gp_i.compute_covariances(ampgs[i], G_gp_i.ls)
            qgl = vmap(
                lambda thi, bi, wi: G_gp_i.compute_q(vgs[i], G_LKvv, thi, bi, wi)
            )(thetagl, betagl, wgl)

            samps += vmap_scan(
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
                        ampgs[i],
                        sigu=u_gp.amp,
                        alpha=self.alpha,
                        pg=G_gp_i.pr,
                        pu=u_gp.pr,
                    )
                )(thetagl, betagl, thetaul, betaul, wgl, qgl, wul, qul,),
                t,
            )

        return samps

    # def _slow_sample(self, t, vgs, vu, ampgs, N_s=10, key=jrnd.PRNGKey(1)):

    #     samps = jnp.zeros((len(t), N_s))
    #     skey = jrnd.split(key, 4)

    #     u_gp = self.u_gp
    #     thetaul = u_gp.sample_thetas(skey[0], (N_s, u_gp.N_basis, 1), u_gp.ls)
    #     betaul = u_gp.sample_betas(skey[1], (N_s, u_gp.N_basis))
    #     wul = 1.0 * u_gp.sample_ws(skey[2], (N_s, u_gp.N_basis), u_gp.amp)

    #     qul = 1.0 * vmap(
    #         lambda thi, bi, wi: u_gp.compute_q(vu, u_gp.LKvv, thi, bi, wi)
    #     )(thetaul, betaul, 1.0 * wul)

    #     for i in range(1, 2):  # 1):

    #         skey = jrnd.split(skey[3], 4)

    #         G_gp_i = self.g_gps[i]
    #         thetagl = G_gp_i.sample_thetas(
    #             skey[0], (N_s, G_gp_i.N_basis, G_gp_i.D), G_gp_i.ls
    #         )
    #         betagl = G_gp_i.sample_betas(skey[1], (N_s, G_gp_i.N_basis))
    #         wgl = 1.0 * G_gp_i.sample_ws(skey[2], (N_s, G_gp_i.N_basis), G_gp_i.amp)
    #         _, G_LKvv = G_gp_i.compute_covariances(ampgs[i], G_gp_i.ls)

    #         qgl = vmap(
    #             lambda thi, bi, wi: G_gp_i.compute_q(vgs[i], G_LKvv, thi, bi, wi)
    #         )(thetagl, betagl, 1.0 * wgl)
    #         # wgl *= 0.0

    #         for k in range(len(t)):
    #             for j in range(N_s):
    #                 print(j)
    #                 a = slow_I(
    #                     t[k],
    #                     G_gp_i.z,
    #                     u_gp.z,
    #                     thetagl[j],
    #                     betagl[j],
    #                     thetaul[j],
    #                     betaul[j],
    #                     wgl[j],
    #                     qgl[j],
    #                     wul[j],
    #                     qul[j],
    #                     ampgs[i],
    #                     sigu=u_gp.amp,
    #                     alpha=self.alpha,
    #                     pg=G_gp_i.pr,
    #                     pu=u_gp.pr,
    #                 )
    #                 print(a)
    #                 samps = index_add(samps, index[k, j], a[0])

    #         return samps

    def sample(self, t, N_s=10, key=jrnd.PRNGKey(1)):
        return self._sample(t, self.vgs, self.vu, self.ampgs, N_s=N_s, key=key)

    def plot_samples(self, t, N_s, save=False, key=jrnd.PRNGKey(1)):
        skey = jrnd.split(key, 2)

        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
        samps = self.sample(t, N_s, key=skey[0])
        axs[0].plot(t, samps, c="green", alpha=0.5)
        axs[0].legend()

        u_samps = self.u_gp.sample(t, N_s, key=skey[1])

        axs[1].plot(t, u_samps, c="blue", alpha=0.5)
        axs[1].scatter(
            self.u_gp.z, self.u_gp.v, label="Inducing Points", marker="x", c="green",
        )
        axs[1].legend()
        if save:
            plt.savefig(save)
        plt.show()


class VariationalNVKM(NVKM):
    def __init__(
        self,
        zgs: List[jnp.DeviceArray],
        zu: jnp.DeviceArray,
        data: Tuple[jnp.DeviceArray, jnp.DeviceArray],
        q_class: VariationalDistribution,
        q_pars_init: Union[VIPars, None] = None,
        q_initializer_pars=None,
        likelihood: Callable = gaussain_likelihood,
        N_basis: int = 500,
        C: int = 1,
        ampgs_init: List[float] = [1.0],
        noise_init: float = 0.5,
        alpha: float = 1.0,
        lsgs: List[float] = [1.0],
        lsu: float = 1.0,
        ampu: float = 1.0,
    ):

        self.zgs = zgs
        self.vgs = [None] * C
        self.zu = zu
        self.vu = None

        self.N_basis = N_basis
        self.C = C
        self.noise = noise_init
        self.alpha = alpha

        self.lsgs = lsgs
        self.lsu = lsu
        self.ampgs = ampgs_init
        self.ampu = ampu

        self.g_gps = self.set_G_gps(ampgs_init, lsgs)
        self.u_gp = self.set_u_gp(ampu, lsu)

        self.p_pars = self._compute_p_pars(self.ampgs, self.lsgs, self.ampu, self.lsu)
        self.data = data
        self.likelihood = likelihood
        self.q_of_v = q_class()
        if q_pars_init is None:
            q_pars_init = self.q_of_v.initialize(self, q_initializer_pars)
        self.q_pars = q_pars_init

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

    @partial(jit, static_argnums=(0,))
    def _compute_p_pars(self, ampgs, lsgs, ampu, lsu):
        return {
            "LK_gs": [
                self.g_gps[i].compute_covariances(ampgs[i], lsgs[i])[1]
                for i in range(self.C)
            ],
            "LK_u": self.u_gp.compute_covariances(ampu, lsu)[1],
        }

    def sample_diag_g_gps(self, ts, N_s, key=jrnd.PRNGKey(1)):
        skey = jrnd.split(key, N_s + 1)
        v_gs = self.q_of_v._sample(self.q_pars, N_s, skey[0])["gs"]
        #         print(v_gs[1][1])
        return [
            vmap(lambda vi, keyi: gp._sample(ts[i], vi, gp.amp, 1, keyi).flatten())(
                v_gs[i], skey[1:]
            ).T
            for i, gp in enumerate(self.g_gps)
        ]

    def sample_u_gp(self, t, N_s, key=jrnd.PRNGKey(1)):
        skey = jrnd.split(key, N_s + 1)

        v_u = self.q_of_v._sample(self.q_pars, N_s, skey[0])["u"]

        return vmap(
            lambda vi, keyi: self.u_gp._sample(
                t, vi, self.u_gp.amp, 1, key=keyi
            ).flatten()
        )(v_u, skey[1:]).T

    @partial(jit, static_argnums=(0, 4))
    def _sample(self, t, q_pars, ampgs, N_s, key=jrnd.PRNGKey(1)):

        skey = jrnd.split(key, 5)
        v_samps = self.q_of_v._sample(q_pars, N_s, skey[4])

        u_gp = self.u_gp
        # print(type((N_s, u_gp.N_basis, 1)))
        thetaul = u_gp.sample_thetas(skey[0], (N_s, u_gp.N_basis, 1), u_gp.ls)
        betaul = u_gp.sample_betas(skey[1], (N_s, u_gp.N_basis))
        wul = u_gp.sample_ws(skey[2], (N_s, u_gp.N_basis), u_gp.amp)

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
            wgl = G_gp_i.sample_ws(skey[2], (N_s, G_gp_i.N_basis), G_gp_i.amp)

            # print(G_gp_i.LKvv, G_Lkvv)
            _, G_LKvv = G_gp_i.compute_covariances(ampgs[i], G_gp_i.ls)
            qgl = vmap(
                lambda vgi, thi, bi, wi: G_gp_i.compute_q(vgi, G_LKvv, thi, bi, wi)
            )(v_samps["gs"][i], thetagl, betagl, wgl)
            # samps += jnp.zeros((len(t), N_s))
            samps += vmap_scan(
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
                        ampgs[i],
                        sigu=u_gp.amp,
                        alpha=self.alpha,
                        pg=G_gp_i.pr,
                        pu=u_gp.pr,
                    )
                )(thetagl, betagl, thetaul, betaul, wgl, qgl, wul, qul,),
                t,
            )

        return samps

    def sample(self, t, N_s, key=jrnd.PRNGKey(1)):
        return self._sample(t, self.q_pars, self.ampgs, N_s, key=key)

    @partial(jit, static_argnums=(0, 5))
    def _compute_bound(self, data, q_pars, ampgs, noise, N_s, key=jrnd.PRNGKey(1)):
        p_pars = self._compute_p_pars(ampgs, self.lsgs, self.ampu, self.lsu)

        KL = self.q_of_v._KL(p_pars, q_pars)

        x, y = data
        samples = self._sample(x, q_pars, ampgs, N_s, key=key)
        like = self.likelihood(y, samples, noise)
        return -(KL + like)

    def compute_bound(self, N_s, key=jrnd.PRNGKey(1)):
        return self._compute_bound(
            self.data, self.q_pars, self.ampgs, self.noise, N_s, key=key
        )

    def fit(self, its, lr, batch_size, N_s, dont_fit=[], key=jrnd.PRNGKey(1)):

        x, y = self.data

        opt_init, opt_update, get_params = opt.adam(lr)

        dpars = {"q_pars": self.q_pars, "noise": self.noise, "ampgs": self.ampgs}
        opt_state = opt_init(dpars)

        for i in range(its):
            skey, key = jrnd.split(key, 2)
            if batch_size:
                rnd_idx = jrnd.choice(key, len(y), shape=(batch_size,))
                y_b = y[rnd_idx]
                x_b = x[rnd_idx]
            else:
                y_b = y
                x_b = x

            # print(get_params(opt_state))
            value, grads = value_and_grad(
                lambda dp: self._compute_bound(
                    (x_b, y_b), dp["q_pars"], dp["ampgs"], dp["noise"], N_s, key=skey
                )
            )(dpars)
            opt_state = opt_update(i, grads, opt_state)

            for k in dpars.keys():
                if k not in dont_fit:
                    dpars[k] = get_params(opt_state)[k]
            #             print(dpars["q_pars"]["mu_u"])
            # this ensurse array are lower triangular
            # should prob go elsewhere
            for j in range(self.C):
                dpars["q_pars"]["LC_gs"][j] = jnp.tril(dpars["q_pars"]["LC_gs"][j])
            dpars["q_pars"]["LC_u"] = jnp.tril(dpars["q_pars"]["LC_u"])

            if jnp.any(jnp.isnan(value)):
                print("nan F!!")
                return get_params(opt_state)

            elif i % 2 == 0:
                print(f"it: {i} F: {value} ")

        # when optimisation complete, set attributes to be new ones
        # and update computations
        for k in dpars.keys():
            setattr(self, k, dpars[k])

        self.p_pars = self._compute_p_pars(self.ampgs, self.lsgs, self.ampu, self.lsu)
        self.g_gps = self.set_G_gps(self.ampgs, self.lsgs)
        self.u_gp = self.set_u_gp(self.ampu, self.lsu)

    def plot_samples(self, t, N_s, save=False, key=jrnd.PRNGKey(13)):
        skey = jrnd.split(key, 2)

        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
        samps = self.sample(t, N_s, key=skey[0])
        axs[0].plot(t, samps, c="green", alpha=0.5)
        axs[0].scatter(*self.data, label="Data", marker="x", c="blue")
        axs[0].legend()

        #         print(self.q_pars["mu_u"])s
        u_samps = self.sample_u_gp(t, N_s, key=skey[1])
        # print(u_samps[0])
        axs[1].plot(t, u_samps, c="blue", alpha=0.5)
        axs[1].scatter(
            self.u_gp.z,
            self.q_pars["mu_u"],
            label="Inducing Points",
            marker="x",
            c="green",
        )
        axs[1].legend()
        if save:
            plt.savefig(save)
        plt.show()

    def plot_filters(self, t, N_s, save=None, key=jrnd.PRNGKey(1)):

        ts = [jnp.vstack((t for i in range(gp.D))).T for gp in self.g_gps]
        g_samps = self.sample_diag_g_gps(ts, N_s, key=key)

        fig, axs = plt.subplots(self.C, 1, figsize=(8, 5 * self.C))
        for i in range(self.C):
            if self.C == 1:
                ax = axs
            else:
                ax = axs[i]
            y = g_samps[i].T * jnp.exp(-self.alpha * (t) ** 2)
            ax.plot(t, y.T, c="red", alpha=0.5)
            ax.set_title(f"$G_{i}$")

        if save:
            plt.savefig(save)
        plt.show()

    def save(self, f_name):
        # with open(f_name, 'wb') as output:  # Overwrites any existing file.
        # pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)
        jnp.savez(f_name, **self.__dict__)
