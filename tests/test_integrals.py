from nvkm import integrals
import jax.numpy as jnp
import jax

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

