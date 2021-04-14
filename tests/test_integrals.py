from nvkm import integrals
import jax.numpy as jnp
import jax.random as jrnd
import jax
import pytest

# all answers from Mathematica exact inegrations
def test_integ_1a():
    ans = jax.lax.complex(-0.18947591534594296, -0.9409343823354436)
    ours = integrals.integ_1a(1.0, 1.2, 1.3, 1.4, 1.5, 1.6)
    assert jnp.isclose(ours, ans)


def test_integ_1b():
    ans = jax.lax.complex(0.41217993289082855, 0.5971748133823835)
    ours = integrals.integ_1b(1.0, 1.2, 1.3, 1.4, 1.5, 1.6)
    assert jnp.isclose(ours, ans)


def test_integ_2a():
    ans = -0.11155473966868128
    ours = integrals.integ_2a(1.0, 1.2, 1.3, 1.4, 1.5, 1.6)
    assert jnp.isclose(ours, ans)


def test_integ_2b():
    ans = 0.04992755433043908
    ours = integrals.integ_2b(1.0, 1.2, 1.3, 1.4, 1.5, 1.6)
    assert jnp.isclose(ours, ans)


@pytest.fixture
def data_maker():
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
        "sigg": 1.0,
    }


def test_I1(data_maker):
    si1 = integrals.slow_I1(
        data_maker["t"],
        data_maker["zus"],
        data_maker["thetags"][0],
        data_maker["betags"][0],
        data_maker["thetus"],
        data_maker["betaus"],
        data_maker["wus"],
        data_maker["qus"],
        data_maker["sigg"],
        sigu=1.1,
        alpha=1.2,
        pu=1.3,
    )
    fi1 = integrals.fast_I1(
        data_maker["t"],
        data_maker["zus"],
        data_maker["thetags"][0],
        data_maker["betags"][0],
        data_maker["thetus"],
        data_maker["betaus"],
        data_maker["wus"],
        data_maker["qus"],
        data_maker["sigg"],
        sigu=1.1,
        alpha=1.2,
        pu=1.3,
    )

    assert jnp.isclose(si1, fi1)


def test_I2(data_maker):
    si2 = integrals.slow_I2(
        data_maker["t"],
        data_maker["zgs"][0],
        data_maker["zus"],
        data_maker["thetus"],
        data_maker["betaus"],
        data_maker["wus"],
        data_maker["qus"],
        data_maker["sigg"],
        sigu=1.1,
        alpha=1.2,
        pg=1.3,
        pu=1.4,
    )
    fi2 = integrals.fast_I2(
        data_maker["t"],
        data_maker["zgs"][0],
        data_maker["zus"],
        data_maker["thetus"],
        data_maker["betaus"],
        data_maker["wus"],
        data_maker["qus"],
        data_maker["sigg"],
        sigu=1.1,
        alpha=1.2,
        pg=1.3,
        pu=1.4,
    )
    assert jnp.isclose(si2, fi2)


def test_I(data_maker):
    si = integrals.slow_I(
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
        data_maker["sigg"],
        sigu=1.1,
        alpha=1.2,
        pg=1.3,
        pu=1.4,
    )
    fi = integrals.fast_I(
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
        data_maker["sigg"],
        sigu=1.1,
        alpha=1.2,
        pg=1.3,
        pu=1.4,
    )
    assert jnp.isclose(si, fi)
