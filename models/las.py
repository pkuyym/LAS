import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers


def blstm(input, unit_size):
    forward_linear_proj = layers.fc(
        input=input, size=unit_size * 4, act=None, bias_attr=False)
    forward_lstm, _ = layers.dynamic_lstm(
        input=forward_linear_proj, size=unit_size * 4)
    backward_linear_proj = layers.fc(
        input=input, size=unit_size * 4, act=None, bias_attr=False)
    backward_lstm, _ = layers.dynamic_lstm(
        input=backward_linear_proj, size=unit_size * 4)
    return layers.concat(input=[forward_lstm, backward_lstm], axis=1)


def plstm(input, unit_size, num_steps=2):
    output = blstm(input, unit_size)
    return layers.sequence_pool(
        input=output, new_dim=input.shape[1] * num_steps)


def listener(input, stacked_num, unit_size, pyramid_steps, dropout_prob,
             is_training):
    '''listener'''
    output = input
    for i in xrange(stacked_num):
        output = plstm(output, unit_size, pyramid_steps)
        # add dropout
        if dropout_prob > 0.0 and is_training:
            output = layers.dropout(output, dropout_prob, not is_training)

    output = blstm(output, unit_size)
    if dropout_prob > 0.0 and is_training:
        output = layers.dropout(output, dropout_prob, not is_training)

    return output
