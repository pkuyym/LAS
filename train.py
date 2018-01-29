import numpy as np
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
from paddle.v2.fluid.executor import Executor
from data_utils.timit_data import DataGenerator
from model_utils.las import listener, speller
from utils.utility import lodtensor_to_ndarray


def adapt_batch_reader(batch_reader, place):
    from data_utils.utility import ndarray_list_to_lodtensor as _to_lodtensor

    def _field(batch, field_idx):
        return map(lambda x: x[field_idx], batch)

    for batch in batch_reader():
        yield {
            'audio_sequence': _to_lodtensor(_field(batch, 0), place),
            'target_sequence': _to_lodtensor(_field(batch, 1), place),
            'target_next_sequence': _to_lodtensor(_field(batch, 2), place),
            'true_token_flag': _to_lodtensor(_field(batch, 3), place),
            'actual_length': _to_lodtensor(_field(batch, 4), place)
        }


cpu_place = core.CPUPlace()
data_path = './data/timit/std_preprocess_26_ch.pkl'
listener_stacked_num = 1
max_label_idx = 61
label_dim = max_label_idx + 2
listener_hidden_dim = 256
pyramid_steps = 2
speller_stacked_num = 1
speller_hidden_dim = 512
dropout_prob = 0.0
batch_size = 4
epoch_num = 2
teacher_force_rate_upperbound = 0.8
teacher_force_rate_lowerbound = 0.0

data_generator = DataGenerator(
    data_path=data_path,
    padding_divisor=2**listener_stacked_num,
    max_idx=max_label_idx)

audio_seq = fluid.layers.data(
    name='audio_sequence',
    shape=[data_generator.feature_dimension],
    dtype='float32',
    lod_level=1)

listener = listener(
    audio_seq=audio_seq,
    stacked_num=listener_stacked_num,
    unit_size=listener_hidden_dim,
    pyramid_steps=pyramid_steps,
    dropout_prob=dropout_prob,
    is_training=True)

trg_word_idx = fluid.layers.data(
    name='target_sequence', shape=[1], dtype='int64', lod_level=1)

true_token_flags = fluid.layers.data(
    name='true_token_flag', shape=[1], dtype='int64', lod_level=1)

speller = speller(
    trg_word_idx=trg_word_idx,
    true_token_flags=true_token_flags,
    stacked_num=speller_stacked_num,
    listener_feature=listener,
    unit_size=speller_hidden_dim,
    label_dim=label_dim,
    is_training=True)

label = fluid.layers.data(
    name='target_next_sequence', shape=[1], dtype='int64', lod_level=1)

loss = layers.cross_entropy(input=speller, label=label)
avg_loss = layers.mean(x=loss)

optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
optimizer.minimize(avg_loss)

place = core.CUDAPlace(0)
exe = Executor(place)
exe.run(framework.default_startup_program())

for epoch_id in xrange(epoch_num):

    teacher_force_rate = teacher_force_rate_upperbound - (
        teacher_force_rate_upperbound - teacher_force_rate_lowerbound) * (
            float(epoch_id) / epoch_num)

    train_batch_reader = data_generator.create_batch_reader(
        batch_size=batch_size,
        is_shuffle=True,
        min_batch_size=4,
        dataset_type='TRAIN',
        teacher_force_rate=teacher_force_rate)

    for batch_id, data in enumerate(
            adapt_batch_reader(train_batch_reader, place)):
        fetch_outs = exe.run(
            framework.default_main_program(),
            feed={
                'audio_sequence': data['audio_sequence'],
                'target_sequence': data['target_sequence'],
                'true_token_flag': data['true_token_flag'],
                'target_next_sequence': data['target_next_sequence']
            },
            fetch_list=[loss, avg_loss],
            return_numpy=False)

        fetch_val, _ = lodtensor_to_ndarray(fetch_outs[1])
        print 'epoch_id=%d, batch_id=%d, avg_loss: %f' % (epoch_id, batch_id,
                                                          fetch_val)
