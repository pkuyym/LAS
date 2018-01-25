import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
from model_utils.layers import blstm, stacked_lstm_unit, pblstm, attention


def listener(audio_seq, stacked_num, unit_size, pyramid_steps, dropout_prob,
             is_training):
    output = audio_seq
    for i in xrange(stacked_num):
        output = pblstm(output, unit_size, pyramid_steps)
        # add dropout
        if dropout_prob > 0.0 and is_training:
            output = layers.dropout(output, dropout_prob, not is_training)

    output = blstm(output, unit_size)
    if dropout_prob > 0.0 and is_training:
        output = layers.dropout(output, dropout_prob, not is_training)

    return output


def speller(stacked_num, listener_feature, unit_size, label_dim):
    trg_word_idx = fluid.layers.data(
        name='target_sequence', shape=[1], dtype='int64', lod_level=1)

    true_token_flags = fluid.layers.data(
        name='true_token_flag', shape=[1], dtype='int64', lod_level=1)

    encoder_last_step = layers.sequence_pool(
        input=listener_feature, pool_type='last')

    # do the projection
    decoder_boot = layers.fc(
        input=encoder_last_step, size=unit_size, bias_attr=False)

    rnn = fluid.layers.DynamicRNN()

    with rnn.block():
        true_word_idx = rnn.step_input(trg_word_idx)
        true_token_flag = rnn.step_input(true_token_flags)
        encoder_vec = rnn.static_input(listener_feature)

        hidden_mem_list = [
            rnn.memory(init=decoder_boot, need_reorder=True)
            for i in xrange(stacked_num)
        ]

        cell_mem_list = [
            rnn.memory(value=0.0, shape=[unit_size])
            for i in xrange(stacked_num)
        ]

        context_mem = rnn.memory(init=encoder_last_step, need_reorder=True)
        out_mem = rnn.memory(value=0, dtype='float32', shape=[label_dim])
        out_mem.stop_gradient = True

        value, generated_word_idx = layers.topk(out_mem, 1)

        cur_word_idx = layers.multiplex(
            inputs=[true_word_idx, generated_word_idx],
            index=layers.cast(x=true_token_flag, dtype='int32'))
        cur_word_idx.stop_gradient = True

        rnn_input = layers.concat(
            [layers.one_hot(input=cur_word_idx, depth=label_dim), context_mem],
            axis=1)

        hidden_t_list, cell_t_list = stacked_lstm_unit(
            rnn_input, hidden_mem_list, cell_mem_list, unit_size)

        for i in xrange(stacked_num):
            rnn.update_memory(hidden_mem_list[i], hidden_t_list[i])
            rnn.update_memory(cell_mem_list[i], cell_t_list[i])

        context = attention(hidden_t_list[-1], encoder_vec)

        rnn.update_memory(context_mem, context)

        concat_feature = layers.concat([hidden_t_list[-1], context], axis=1)

        out = fluid.layers.fc(
            input=concat_feature, size=label_dim, bias_attr=True, act='softmax')
        rnn.update_memory(out_mem, out)
        rnn.output(out)

    decoder_output = rnn()

    return decoder_output
