import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers


def blstm(input, unit_size):
    """ Bidirectional lstm layer. This layer contains two lstm units, one for
    forward computation and the other for backward computation. Please note,
    weights of the two lstm units are not shared.

        Args:
            input (Variable): Input sequence which is a LoDTensor.
            unit_size (int): Unit size of lstm.

        Returns:
            Variable: Concated output of the two lstm units.
    """
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


def stacked_lstm_unit(x_t, hidden_t_prev_list, cell_t_prev_list, unit_size):
    assert len(hidden_t_prev_list) == len(cell_t_prev_list)
    stacked_num = len(hidden_t_prev_list)
    hidden_t_list = []
    cell_t_list = []
    for i in xrange(stacked_num):
        hidden_t, cell_t = lstm_unit(
            x_t=x_t,
            hidden_t_prev=hidden_t_prev_list[i],
            cell_t_prev=cell_t_prev_list[i],
            unit_size=unit_size)
        hidden_t_list.append(hidden_t)
        cell_t_list.append(cell_t)
        x_t = hidden_t
    return hidden_t_list, cell_t_list


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
