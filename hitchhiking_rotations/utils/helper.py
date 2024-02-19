def passthrough(x):
    return x


def flatten(x):
    return x.reshape(x.shape[0], -1)
