import numpy as np

def moving_average(x, N):
    if len(x) == 0:
        return []
    remainder = len(x) % N
    y = x[:len(x) - remainder]
    y = np.reshape(y, (-1, N))
    y = np.mean(y, axis=1)
    y = y.reshape(-1)
    return y