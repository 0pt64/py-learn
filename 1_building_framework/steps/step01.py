import numpy as np


class Variable:
    def __init__(self, data) -> None:
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = x ** 2
        output = Variable(y)
        return output


data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)

x = np.array(1)
print(x.ndim)

x = np.array([1, 2, 3])
print(x.ndim)

x = np.array([[1, 2, 3], [4, 5, 6]])
print(x.ndim)

x = Variable(np.array(10))
f = Function()
y = f(x)

print(type(y))
print(y.data)
