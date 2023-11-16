import torch.nn as nn
from tencentpretrain import mpu
from tencentpretrain.utils import *

class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer. """
    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = str2act[hidden_act]

    def forward(self, x):
        inter = self.act(self.linear_1(x))
        output = self.linear_2(inter)
        return output


class GatedFeedForward(nn.Module):
    """ Feed Forward Layer with Gated Linear Unit.
        https://arxiv.org/abs/2002.05202
    """
    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(GatedFeedForward, self).__init__()
        self.linear_gate = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = str2act[hidden_act]

    def forward(self, x):
        gate = self.act(self.linear_gate(x))
        inter_linear = self.linear_1(x)
        inter = gate * inter_linear
        output = self.linear_2(inter)

        return output


class ParallelPositionwiseFeedForward(nn.Module):

    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(ParallelPositionwiseFeedForward, self).__init__()
        self.linear_1 = mpu.ColumnParallelLinear(hidden_size, feedforward_size, gather_output=False, skip_bias_add= False if has_bias else True)
        self.linear_2 = mpu.RowParallelLinear(feedforward_size, hidden_size, skip_bias_add= False if has_bias else True, input_is_parallel=True)
        self.act = str2act[hidden_act]

    def forward(self, hidden_states):
        # [s, b, h] -> [s, b, 4h]
        intermediate_parallel = self.linear_1(hidden_states)
        intermediate_parallel = self.act(intermediate_parallel)
        # [s, b, 4h] -> [s, b, h]
        output = self.linear_2(intermediate_parallel)
        return output

class ParallelGatedFeedForward(nn.Module):

    def __init__(self, hidden_size, feedforward_size, hidden_act, has_bias=True):
        super(ParallelGatedFeedForward, self).__init__()
        self.linear_gate = mpu.ColumnParallelLinear(hidden_size, feedforward_size, gather_output=False, skip_bias_add= False if has_bias else True)
        self.linear_1 = mpu.ColumnParallelLinear(hidden_size, feedforward_size, gather_output=False, skip_bias_add= False if has_bias else True)
        self.linear_2 = mpu.RowParallelLinear(feedforward_size, hidden_size, skip_bias_add= False if has_bias else True, input_is_parallel=True)
        self.act = str2act[hidden_act]
    def forward(self, hidden_states):
        # [s, b, h] -> [s, b, 4h]
        intermediate_parallel = self.linear_gate(hidden_states)
        intermediate_parallel = self.act(intermediate_parallel)
        intermediate_parallel=intermediate_parallel*self.linear_1(hidden_states)
        # [s, b, 4h] -> [s, b, h]
        output = self.linear_2(intermediate_parallel)
        return output
