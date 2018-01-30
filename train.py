import numpy as np
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
from paddle.v2.fluid.executor import Executor
from data_utils.timit_data import DataGenerator
from model_utils.las import listener, speller
from utils.utility import lodtensor_to_ndarray
from config.las_example_config import cfg


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


data_generator = DataGenerator(
    data_path=cfg.data.path,
    padding_divisor=2**cfg.net.listener_stacked_num,
    max_idx=cfg.data.max_label_idx)

audio_seq = fluid.layers.data(
    name='audio_sequence',
    shape=[data_generator.feature_dimension],
    dtype='float32',
    lod_level=1)

listener = listener(
    audio_seq=audio_seq,
    stacked_num=cfg.net.listener_stacked_num,
    unit_size=cfg.net.listener_hidden_dim,
    pyramid_steps=cfg.net.pyramid_steps,
    dropout_prob=cfg.net.dropout_prob,
    is_training=True)

trg_word_idx = fluid.layers.data(
    name='target_sequence', shape=[1], dtype='int64', lod_level=1)

true_token_flags = fluid.layers.data(
    name='true_token_flag', shape=[1], dtype='int64', lod_level=1)

speller = speller(
    trg_word_idx=trg_word_idx,
    true_token_flags=true_token_flags,
    stacked_num=cfg.net.speller_stacked_num,
    listener_feature=listener,
    unit_size=cfg.net.speller_hidden_dim,
    label_dim=cfg.net.label_dim,
    is_training=True)

label = fluid.layers.data(
    name='target_next_sequence', shape=[1], dtype='int64', lod_level=1)

loss = layers.cross_entropy(input=speller, label=label)
avg_loss = layers.mean(x=loss)

optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
optimizer.minimize(avg_loss)

place = core.CPUPlace() if cfg.device == 'CPU' else core.CUDAPlace(0)
exe = Executor(place)
exe.run(framework.default_startup_program())

for epoch_id in xrange(cfg.train.epoch_num):
    teacher_force_rate = cfg.train.teacher_force_rate_upperbound - (
        cfg.train.teacher_force_rate_upperbound -
        cfg.train.teacher_force_rate_lowerbound) * (float(epoch_id) /
                                                    cfg.train.epoch_num)

    train_batch_reader = data_generator.create_batch_reader(
        batch_size=cfg.train.batch_size,
        is_shuffle=True,
        min_batch_size=cfg.train.min_batch_size,
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
