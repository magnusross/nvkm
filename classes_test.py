class ClassA:
    def __init__(self, a, b, d=1.0):
        self.a = a
        self.b = b
        self.d = d

    def sq(self, x):
        return self.a * x ** 2

    def lin(self, x):
        return self.b * x


class ClassB(ClassA):
    def __init__(self, a, b, c, d=3.0):
        super().__init__(a, b, d=d)
        self.c = c

    def sq(self, x):
        return (self.a + self.b) * x ** 2

    def cb(self, x):
        return self.c * x ** 3


ca = ClassA(3.0, 4.0)
cb = ClassB(3.0, 4.0, 5.0,)

