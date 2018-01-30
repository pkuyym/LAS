from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

__C.data = edict()
__C.data.path = './data/timit/std_preprocess_26_ch.pkl'
__C.data.max_label_idx = 61

__C.device = 'GPU'

__C.net = edict()
__C.net.listener_stacked_num = 3
__C.net.listener_hidden_dim = 256
__C.net.pyramid_steps = 2
__C.net.speller_stacked_num = 1
__C.net.speller_hidden_dim = 512
__C.net.dropout_prob = 0.0
__C.net.label_dim = __C.data.max_label_idx + 2

__C.train = edict()
__C.train.batch_size = 4
__C.train.epoch_num = 2
__C.train.teacher_force_rate_upperbound = 0.8
__C.train.teacher_force_rate_lowerbound = 0.0
__C.train.min_batch_size = 4
