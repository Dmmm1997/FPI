import numpy as np


def vector2array(vector):
    n, p, c = vector.shape
    h = w = np.sqrt(p)
    if int(h) * int(w) != int(p):
        raise ValueError("p can not be sqrt")
    else:
        h = int(h)
        w = int(w)
    array = vector.permute(0, 2, 1).contiguous().view(n, c, h, w)
    return array