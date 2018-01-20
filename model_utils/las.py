import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers


def blstm(input, unit_size):
    forward_linear_proj = layers.fc(
        input=input, size=unit_size * 4, act=None, bias_attr=False)
    forward_lstm, _ = layers.dynamic_lstm(
        input=forward_linear_proj, size=unit_size * 4, use_peepholes=False)
    backward_linear_proj = layers.fc(
        input=input, size=unit_size * 4, act=None, bias_attr=False)
    backward_lstm, _ = layers.dynamic_lstm(
        input=backward_linear_proj, size=unit_size * 4, use_peepholes=False)
    return layers.concat(input=[forward_lstm, backward_lstm], axis=1)


def pblstm(input, unit_size, num_steps=2):
    output = blstm(input, unit_size)
    return layers.sequence_reshape(
        input=output, new_dim=output.shape[1] * num_steps)


def listener(audio_seq, stacked_num, unit_size, pyramid_steps, dropout_prob,
             is_training):
    '''listener'''
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


def stacked_lstm_unit(x_t, stacked_num, hidden_t_prev_list, cell_t_prev_list,
                      unit_size):
    def lstm_unit(x_t, hidden_t_prev, cell_t_prev, unit_size):
        def linear(inputs):
            return layers.fc(input=inputs, size=unit_size, bias_attr=True)

        forget_gate = layers.sigmoid(x=linear([hidden_t_prev, x_t]))
        input_gate = layers.sigmoid(x=linear([hidden_t_prev, x_t]))
        output_gate = layers.sigmoid(x=linear([hidden_t_prev, x_t]))
        cell_tilde = layers.tanh(x=linear([hidden_t_prev, x_t]))

        cell_t = layers.sums(input=[
            layers.elementwise_mul(x=forget_gate, y=cell_t_prev),
            layers.elementwise_mul(x=input_gate, y=cell_tilde)
        ])

        hidden_t = layers.elementwise_mul(
            x=output_gate, y=fluid.layers.tanh(x=cell_t))

        return hidden_t, cell_t

    assert stacked_num == len(hidden_t_prev_list)
    assert stacked_num == len(cell_t_prev_list)

    hidden_t_list = []
    cell_t_list = []
    for i in xrange(stacked_num):
        hidden_t, cell_t = lstm_unit(
            x_t=x_t, hidden_t_prev_list[i], cell_t_prev_list[i], unit_size)
        hidden_t_list.append(hidden_t)
        cell_t_list.append(cell_t)
        x_t = hidden_t
    return hidden_t_list, cell_t_list


def speller(stacked_num, listener_feature, unit_size, label_dim):
    ''' currently only support one layer '''

    def attention(decoder_state, encoder_vec):
        decoder_state_expand = layers.sequence_expand(
            x=decoder_state, y=encoder_vec)
        attention_weights = layers.fc(
            input=[decoder_state_expand, encoder_vec], size=1, bias_attr=False)
        attention_weights = layers.sequence_softmax(x=attention_weights)
        weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=weigths_reshape, axis=0)
        context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        return context

    trg_word_idx = fluid.layers.data(
        name='target_sequence', shape=[1], dtype='int64', lod_level=1)

    true_token_flags = paddle.layers.data(
        name='true_token_flag', shape=[1], dtype='int64', lod_level=1)

    decoder_boot = layers.sequence_pool(
        input=listener_feature, pool_type='last')

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

        # no embedding just one-hot ids
        out_mem = rnn.memory(value=0, dtype='int64', shape=[label_dim])

        value, generated_word_idx = layers.topk(out_mem, 1)

        # @TODO to be added
        current_word = layers.multiplex([
            true_word_idx,
            generated_word_idx,
        ], true_token_flag)
        # @TODO to be added
        # one-hot

        if rnn_input is None:
            # make sure dims[1] of listener_feature.last_timestep
            # equal to unit_size
            if decoder_boot.shape[1] != unit_size:
                # do projection
                pass
            rnn_input = layers.concat(
                [one - hot(current_word), decoder_boot], axis=1)
        else:
            rnn_input = layers.concat(
                [one - hot(current_word), context], axis=1)

        hidden_t_list, cell_t_list = stacked_lstm_unit(
            rnn_input, hidden_mem_list, cell_mem_list, unit_size)

        for i in xrange(stacked_num):
            rnn.update_memory(hidden_mem_list[i], hidden_t_list[i])
            rnn.update_memory(cell_mem_list[i], cell_t_list[i])

        context = attention(hidden_t_list[-1], encoder_vec)
        concat_feature = concat([hidden_t_list[-1], context])
        out = fluid.layers.fc(
            input=concat_feature, size=label_dim, bias_attr=True, act='softmax')
        rnn.update_memory(out_mem, out)
        rnn.output(out)

    return rnn()
