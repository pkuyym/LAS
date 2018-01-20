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


def listener(input, stacked_num, unit_size, pyramid_steps, dropout_prob,
             is_training):
    '''listener'''
    output = input
    for i in xrange(stacked_num):
        output = pblstm(output, unit_size, pyramid_steps)
        # add dropout
        if dropout_prob > 0.0 and is_training:
            output = layers.dropout(output, dropout_prob, not is_training)

    output = blstm(output, unit_size)
    if dropout_prob > 0.0 and is_training:
        output = layers.dropout(output, dropout_prob, not is_training)

    return output


def lstm_step(x_t, hidden_t_prev, cell_t_prev, unit_size):
    def linear(inputs):
        return fluid.layers.fc(input=inputs, size=unit_size, bias_attr=True)

    forget_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    input_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    output_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    cell_tilde = fluid.layers.tanh(x=linear([hidden_t_prev, x_t]))

    cell_t = fluid.layers.sums(input=[
        fluid.layers.elementwise_mul(x=forget_gate, y=cell_t_prev),
        fluid.layers.elementwise_mul(x=input_gate, y=cell_tilde)
    ])

    hidden_t = fluid.layers.elementwise_mul(
        x=output_gate, y=fluid.layers.tanh(x=cell_t))

    return hidden_t, cell_t


def attention(input):
    pass


def speller(input, listener_feature, unit_size, label_dim):
    pass
    rnn = fluid.layers.DynamicRNN()

    with rnn.block():
        # may consider the multiplex
        current_word = rnn.step_input(input)
        encoder_vec = rnn.static_input(listener_feature)
        hidden_mem = rnn.memory(value=0.0, shape=[unit_size])
        cell_mem = rnn.memory(value=0.0, shape=[unit_size])
        # do attention

        # compute the output

    return rnn()
