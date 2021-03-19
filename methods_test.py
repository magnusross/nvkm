#%%
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt
from functools import partial

#%%
    
class Tester1:
    def __init__(self, xinit=10.0):
        self.x = xinit

    def f(self, x=None):
        if not x:
            x = self.x
        return (x - 3.0) * (x - 10.0) + 1000 * jnp.sin(x)

    def opt_f(self):
        for i in range(100):
            self.x -= 0.1 * 0.001 * grad(self.f)(self.x)


class Tester2:
    def __init__(self, xinit=10.0):
        self.x = xinit

    @partial(jit, static_argnums=(0,))
    def f(self, x):
        return (x - 3.0) * (x - 10.0) + 1000 * jnp.sin(x)

    def opt_f(self):
        for i in range(100):
            self.x -= 0.1 * 0.001 * grad(self.f)(self.x)

@jit
def f(x):
    return (x - 3.0) * (x - 10.0) + 1000 * jnp.sin(x)

def opt_f(xinit):
    x = xinit
    for i in range(100):
        x -= 0.1 * 0.001 * grad(f)(x)
    return x 


test1 = Tester1()
%timeit test1.opt_f()
test2 = Tester2()
%timeit test2.opt_f()
%timeit opt_f(10.0)

t = jnp.linspace(-5, 50)
plt.plot(t, (t - 3.0) * (t - 10.0) + 1000 * jnp.sin(t))
plt.vlines(test.x, 0.0, 1000.0)
plt.show()
#%%
def method(cls):
    """Decorator to add the function as a method to a class.
    Args:
        cls (type): Class to add the function as a method to.
    """

    def decorator(f):
        setattr(cls, f.__name__, f)
        return f

    return decorator

class Tester4:
    def __init__(self, xinit=10.0):
        self.x = xinit
        self.N = 5
    def opt_f(self):
        for i in range(100):
            self.x -= 0.1 * 0.001 * grad(self.f)(self.x)



@method(Tester4)
@partial(jit, static_argnums=(0,))
def f(model, x):
    return model.N*(x - 3.0) * (x - 10.0) + 1000 * jnp.sin(x)
# %%
test = Tester4()
%timeit test.opt_f()
# %%
