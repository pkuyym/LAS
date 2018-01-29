import numpy as np


def lodtensor_to_ndarray(lod_tensor):
    dims = lod_tensor.get_dims()
    ret = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ret.ravel()[i] = lod_tensor.get_float_element(i)
    return ret, lod_tensor.lod()
