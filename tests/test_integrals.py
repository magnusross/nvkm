from nvkm.integrals import Full, Separable
import jax.numpy as jnp
import jax.random as jrnd
import jax
import pytest

# all answers from Mathematica exact inegrations
class TestFull:
    def test_integ_1a(self):
        ans = jax.lax.complex(-0.9558973454319415, 0.15076667809523533)
        ours = Full.integ_1a(1.0, 1.2, 1.3, 1.4, 1.5)
        assert jnp.isclose(ours, ans)

    def test_integ_1b(self):
        ans = jax.lax.complex(0.7467893006351713, -0.27259937436632203)
        ours = Full.integ_1b(1.0, 1.2, 1.3, 1.4, 1.5)
        assert jnp.isclose(ours, ans)

    def test_integ_2a(self):
        ans = -0.11155473966868128
        ours = Full.integ_2a(1.0, 1.2, 1.3, 1.4, 1.5, 1.6)
        assert jnp.isclose(ours, ans)

    def test_integ_2b(self):
        ans = 0.04992755433043908
        ours = Full.integ_2b(1.0, 1.2, 1.3, 1.4, 1.5, 1.6)
        assert jnp.isclose(ours, ans)

    @pytest.fixture
    def data_maker(self):
        key = jrnd.PRNGKey(10)
        keys = jrnd.split(key, 8)
        return {
            "t": 0.1,
            "zgs": jrnd.uniform(key, shape=(5, 3)),
            "zus": jnp.linspace(-1.0, 1.0, 5),
            "thetags": jrnd.normal(keys[0], shape=(8, 3)),
            "betags": jrnd.uniform(keys[1], shape=(8,)),
            "thetus": jrnd.normal(keys[2], shape=(5,)),
            "betaus": jrnd.uniform(keys[3], shape=(5,)),
            "wgs": jrnd.normal(keys[4], shape=(8,)),
            "qgs": jrnd.normal(keys[5], shape=(5,)),
            "wus": jrnd.normal(keys[6], shape=(5,)),
            "qus": jrnd.normal(keys[7], shape=(5,)),
        }

    def test_I1(self, data_maker):
        si1 = Full.slow_I1(
            data_maker["t"],
            data_maker["zus"],
            data_maker["thetags"][0],
            data_maker["betags"][0],
            data_maker["thetus"],
            data_maker["betaus"],
            data_maker["wus"],
            data_maker["qus"],
            1.2,
            1.3,
        )
        fi1 = Full.fast_I1(
            data_maker["t"],
            data_maker["zus"],
            data_maker["thetags"][0],
            data_maker["betags"][0],
            data_maker["thetus"],
            data_maker["betaus"],
            data_maker["wus"],
            data_maker["qus"],
            1.2,
            1.3,
        )

        assert jnp.isclose(si1, fi1)

    def test_I2(self, data_maker):
        si2 = Full.slow_I2(
            data_maker["t"],
            data_maker["zgs"][0],
            data_maker["zus"],
            data_maker["thetus"],
            data_maker["betaus"],
            data_maker["wus"],
            data_maker["qus"],
            1.2,
            1.3,
            1.4,
        )
        fi2 = Full.fast_I2(
            data_maker["t"],
            data_maker["zgs"][0],
            data_maker["zus"],
            data_maker["thetus"],
            data_maker["betaus"],
            data_maker["wus"],
            data_maker["qus"],
            1.2,
            1.3,
            1.4,
        )
        assert jnp.isclose(si2, fi2)

    def test_I(self, data_maker):
        si = Full.slow_I(
            data_maker["t"],
            data_maker["zgs"],
            data_maker["zus"],
            data_maker["thetags"],
            data_maker["betags"],
            data_maker["thetus"],
            data_maker["betaus"],
            data_maker["wgs"],
            data_maker["qgs"],
            data_maker["wus"],
            data_maker["qus"],
            1.2,
            1.3,
            1.4,
        )
        fi = Full.fast_I(
            data_maker["t"],
            data_maker["zgs"],
            data_maker["zus"],
            data_maker["thetags"],
            data_maker["betags"],
            data_maker["thetus"],
            data_maker["betaus"],
            data_maker["wgs"],
            data_maker["qgs"],
            data_maker["wus"],
            data_maker["qus"],
            1.2,
            1.3,
            1.4,
        )
        assert jnp.isclose(si, fi)


# 1.8723126229503788
class TestSeparable:
    def test_I_phi_phi(self):
        ours = Separable.I_phi_phi(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        ans = 1.8807937528073269
        assert jnp.isclose(ans, ours)

    def test_I_k_phi(self):
        ours = Separable.I_k_phi(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        ans = 1.8723126229503788
        assert jnp.isclose(ans, ours)
