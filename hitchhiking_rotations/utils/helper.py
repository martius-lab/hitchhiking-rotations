def passthrough(*x):
    if len(x) == 1:
        return x[0]
    return x


def flatten(x):
    return x.reshape(x.shape[0], -1)
