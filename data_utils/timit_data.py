from six.moves import cPickle
import random
import numpy as np


class DataGenerator(object):
    def __init__(self,
                 data_path,
                 padding_divisor=8,
                 max_idx=61,
                 random_seed=0,
                 audio_dtype='float32',
                 label_dtype='int64'):
        self._rng = random.Random(random_seed)
        np.random.seed(random_seed)
        with open(data_path, 'rb') as cPickle_file:
            [
                self._X_train, self._y_train, self._X_val, self._y_val,
                self._X_test, self._y_test
            ] = cPickle.load(cPickle_file)
        self._feature_dimension = self._X_train[0].shape[1]
        self._max_idx = max_idx  # 0 and 1 are reserved
        self._audio_dtype = audio_dtype
        self._label_dtype = label_dtype
        self._padding_divisor = padding_divisor

    def create_batch_reader(self,
                            batch_size,
                            is_shuffle,
                            min_batch_size,
                            dataset_type,
                            teacher_force_rate=0.9):
        audio_data = None
        label_data = None

        if dataset_type == 'TRAIN':
            audio_data = self._X_train
            label_data = self._y_train
        elif dataset_type == 'VAL':
            audio_data = self._X_val
            label_data = self._y_val
        elif dataset_type == 'TEST':
            audio_data = self._X_test
            label_data = self._y_test
        else:
            raise ValueError("Unsupported dataset type. "
                             "Should be 'TRAIN', 'VAL' or 'TEST'.")

        if is_shuffle == True:
            index_list = range(0, len(audio_data))
            self._rng.shuffle(index_list)
            audio_data = [audio_data[i] for i in index_list]
            label_data = [label_data[i] for i in index_list]

        def batch_reader():
            batch = []
            for i, data in enumerate(audio_data):
                padded_data, actual_len = self._padding_data(
                    data, self._padding_divisor)
                label_seq = self._process_label(label_data[i])
                true_label_flags = self._generate_true_label_flags(
                    label_seq, teacher_force_rate)
                batch.append(
                    (padded_data, label_seq, true_label_flags, actual_len))
                if len(batch) == batch_size:
                    yield batch
                    batch = []

            if len(batch) >= min_batch_size:
                yield batch

        return batch_reader

    def _padding_data(self, data, padding_divisor):
        actual_len = data.shape[0]
        pad_len = 0
        if actual_len % padding_divisor != 0:
            pad_len = padding_divisor - (actual_len % padding_divisor)
        padded_data = np.zeros(
            (actual_len + pad_len, data.shape[1])).astype(self._audio_dtype)
        padded_data[:actual_len, :] = data
        return padded_data, np.array([actual_len]).astype(self._label_dtype)

    def _process_label(self, label_data):
        processed_label_data = []
        last_label = -1
        for label in label_data:
            if label != last_label:
                processed_label_data.append(label)
                last_label = label
        processed_label_data.append(1)  # make label end with <eos>
        return np.array(processed_label_data).astype(self._label_dtype)

    def _generate_true_label_flags(self, label_seq, teacher_force_rate):
        true_label_flags = np.ones(label_seq.shape).astype(self._label_dtype)
        for i, label in enumerate(label_seq):
            if np.random.random_sample() >= teacher_force_rate:
                true_label_flags[i] = 0
        return true_label_flags

    @property
    def feature_dimension(self):
        return self._feature_dimension
