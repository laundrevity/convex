import convex
import numpy as np
import time


if __name__ == '__main__':
    Q = np.array([[1, 0], [0, 1]]).flatten()
    p = np.array([0, 0])
    b = np.array([1, 1])
    s = np.array([1, 1])
    a = np.array([1, 1])
    E = 1
    x0 = np.array([0.0, 0.0])

    # dynamic step size params
    alpha = 1.0
    gamma = 1.0
    max_iter = 25

    t0 = time.time()
    x_star = convex.optimize(Q, p, b, s, x0, max_iter, a, E, alpha, gamma)
    t1 = time.time()

    print(f"x_star: {x_star}")
    print(f'calculation time: {(t1-t0)*1000:.2f} ms')
