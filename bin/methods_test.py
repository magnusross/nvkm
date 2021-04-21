#%%
import jax.numpy as jnp
import jax.random as jrnd
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

class Tester5:
    def __init__(self):
        self.x = 10.

class Tester6:
    def __init__(self):
        pass

    @partial(jit, static_argnums=(0,))
    def do_x(self, t: Tester5):
        return t.x **2
        
t5 = Tester5()
t6 = Tester6()
t6.do_x(t5)       
# %%
@jit
def s(x, key):
    return x * jrnd.uniform(key)

# %%
s(1, jrnd.PRNGKey(1))
# %%

class Tester7:
    def __init__(self, a):
        self.a = a 

    # @partial(jit, static_argnums=(0,))
    def f(self, ts):
        return [self.a[i] * ti ** 2 for i, ti in enumerate(ts)]

class Tester8(Tester7):
    def __init__(self, a):
        super().__init__([a])

    # @partial(jit, static_argnums=(0,))
    def f(self, t):
        return super().f([t])[0]

print(Tester8(10.).f(5.))
Tester7([10.0, 10.0]).f([5., 5.])
# %%
