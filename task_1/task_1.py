import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


data = np.genfromtxt('task_1_capital.txt', skip_header=1, dtype=np.int)
X, y = np.split(data, indices_or_sections=2, axis=1)
samples_amt = len(X)
max_x = max(X)

X = np.append(
    X,
    np.ones((samples_amt, 1), dtype=np.int),
    axis=1
)

b = inv(X.T.dot(X)).dot(X.T.dot(y))

plt.title("Rental regressed on Capital")
plt.xlabel("Capital")
plt.ylabel("Rental")
plt.scatter(
    np.delete(X, -1, axis=1),
    y,
    color="red",
    s=5
)

x = np.linspace(0, max_x, 500)
plt.plot(
    [0, max_x],
    [b[1], b[0] * max_x + b[1]],
    color="blue"
)
plt.show()
plt.savefig('output.png')