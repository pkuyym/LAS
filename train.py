import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
from data_utils.timit_data import DataGenerator
from mode_utils.las import listener, speller


def adapt_batch_reader(batch_reader, place):
    def to_lodtensor(data):
        fea_dim = data[0].shape[1] if len(data[0].shape) == 2 else 1
        dtype = data[0].dtype
        lod = [[0]]
        for seq in data:
            lod[0].append(lod[0][-1] + seq.shape[0])
        shape = (lod[0][-1], fea_dim)
        np_data = np.zeros(
            np.product(shape), ).astype(dtype)
        print lod
        print shape
        for i, seq in enumerate(data):
            np_data[lod[0][i] * fea_dim:lod[0][i + 1] * fea_dim] = seq.flatten()

        lod_tensor = core.LoDTensor()
        lod_tensor.set(np_data, place)
        lod_tensor.set_lod(lod)

        return lod_tensor

    for batch in batch_reader():
        yield {
            'audio_sequence': to_lodtensor([ins[0] for ins in batch]),
            'target_sequence': to_lodtensor([ins[1] for ins in batch]),
            'true_token_flag': to_lodtensor([ins[2] for ins in batch]),
            'actual_length': to_lodtensor([ins[3] for ins in batch])
        }


cpu_place = core.CPUPlace()
data_path = './data/timit/std_preprocess_26_ch.pkl'
listener_stacked_num = 3
max_label_idx = 61
label_dim = max_label_idx + 2

data_generator = DataGenerator(
    data_path=data_path,
    padding_divisor=2**listener_stacked_num,
    max_idx=max_label_idx)

train_batch_reader = adapt_batch_reader(data_generator.create_batch_reader())
'''
audio_seq = fluid.layers.data(name='audio_sequence',
                              shape=[data_generator.feature_dimension],
                              dtype='float32',
                              lod_level=1)

trg_word_idx = fluid.layers.data(name='target_sequence',
                                 shape=[1],
                                 dtype='int64',
                                 lod_level=1)

true_token_flags = fluid.layers.data(name='true_token_flag',
                                     shape=[1],
                                     dtype='int32',
                                     lod_level=1)

listener()

speller()


'''
