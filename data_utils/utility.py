import numpy as np
import paddle.v2.fluid.core as core


def ndarray_list_to_lodtensor(data, place):
    feature_dimension = data[0].shape[1] if len(data[0].shape) == 2 else 1
    dtype = data[0].dtype
    lod = [[0]]
    for seq in data:
        lod[0].append(lod[0][-1] + seq.shape[0])
    shape = (lod[0][-1], feature_dimension)
    np_data = np.zeros(shape).astype(dtype)

    for i, seq in enumerate(data):
        np_data[lod[0][i]:lod[0][i + 1], :] = seq.reshape(
            (-1, feature_dimension))

    lod_tensor = core.LoDTensor()
    lod_tensor.set(np_data, place)
    lod_tensor.set_lod(lod)

    return lod_tensor
