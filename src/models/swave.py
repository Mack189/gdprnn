
import functools
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer, HeNormal


class MulCatBlock(nn.Cell):
    def __init__(self, input_size, hidden_size, dropout=0.0, bidirectional=False):
        super(MulCatBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = nn.LSTM(input_size, hidden_size, 1, dropout=dropout,
                           batch_first=True, bidirectional=bidirectional)
        self.rnn_proj = nn.Dense(hidden_size * self.num_direction, input_size, weight_init="HeNormal")

        self.gate_rnn = nn.LSTM(input_size, hidden_size, num_layers=1,
                                batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.gate_rnn_proj = nn.Dense(
            hidden_size * self.num_direction, input_size, weight_init="HeNormal")

        self.block_projection = nn.Dense(input_size * 2, input_size, weight_init="HeNormal")
        self.mul = ops.Mul()
        self.op2 = ops.Concat(2)

    def construct(self, inputs):
        output = inputs
        rnn_output, _ = self.rnn(output)
        rnn_output = self.rnn_proj(rnn_output.view(-1, rnn_output.shape[2])).view(output.shape)
        # run gate rnn module
        gate_rnn_output, _ = self.gate_rnn(output)
        gate_rnn_output = self.gate_rnn_proj(gate_rnn_output.view(-1, gate_rnn_output.shape[2])).view(output.shape)
        # apply gated rnn
        gated_output = self.mul(rnn_output, gate_rnn_output)
        gated_output = self.op2([gated_output, output])
        gated_output = self.block_projection(
            gated_output.view(-1, gated_output.shape[2])).view(output.shape)
        return gated_output


class ByPass(nn.Cell):
    def construct(self, inputs):
        return inputs


class DPMulCat(nn.Cell):
    def __init__(self, input_size, hidden_size, output_size, num_spk,
                 dropout=0.0, num_layers=1, bidirectional=True, input_normalize=False):
        super(DPMulCat, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.in_norm = input_normalize
        self.num_layers = num_layers

        self.rows_grnn = nn.CellList([])
        self.cols_grnn = nn.CellList([])
        self.rows_normalization = nn.CellList([])
        self.cols_normalization = nn.CellList([])

        # create the dual path pipeline
        for _ in range(num_layers):
            self.rows_grnn.append(MulCatBlock(
                input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.cols_grnn.append(MulCatBlock(
                input_size, hidden_size, dropout, bidirectional=bidirectional))
            if self.in_norm:
                self.rows_normalization.append(
                    nn.GroupNorm(1, input_size, eps=1e-8))
                self.cols_normalization.append(
                    nn.GroupNorm(1, input_size, eps=1e-8))
            else:
                # used to disable normalization
                self.rows_normalization.append(ByPass())
                self.cols_normalization.append(ByPass())
        self.output = nn.SequentialCell(
            nn.PReLU(), nn.Conv2d(input_size, output_size * num_spk, 1, has_bias=True))

    def construct(self, inputs):
        batch_size, _, d1, d2 = inputs.shape
        output = inputs
        output_all = []
        for i in range(self.num_layers):
            row_input = output.transpose(0, 3, 2, 1).view(
                batch_size * d2, d1, -1)
            row_output = self.rows_grnn[i](row_input)
            row_output = row_output.view(
                batch_size, d2, d1, -1).transpose(0, 3, 2, 1)
            row_output = self.rows_normalization[i](row_output)
            # apply a skip connection
            if not self.training:
                output = output + row_output
            else:
                output += row_output

            col_input = output.transpose(0, 2, 3, 1).view(
                batch_size * d1, d2, -1)
            col_output = self.cols_grnn[i](col_input)
            col_output = col_output.view(
                batch_size, d1, d2, -1).transpose(0, 3, 1, 2)
            col_output = self.cols_normalization[i](col_output)
            # apply a skip connection
            if not self.training:
                output = output + col_output
            else:
                output += col_output

            output_i = self.output(output)
            if not self.training or i == (self.num_layers - 1):
                output_all.append(output_i)
        return output_all

class Separator(nn.Cell):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, num_spk=2,
                 layer=4, segment_size=100, input_normalize=False, bidirectional=True):
        super(Separator, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.input_normalize = input_normalize
        self.zeros = ops.Zeros()
        self.op2 = ops.Concat(2)
        self.op3 = ops.Concat(3)
        self.transpose = ops.Transpose()
        self.input_perm = (0, 1, 3, 2)

        self.rnn_model = DPMulCat(self.feature_dim, self.hidden_dim,
                                  self.feature_dim, self.num_spk,
                                  num_layers=layer, bidirectional=bidirectional, input_normalize=input_normalize)

    def pad_segment(self, inputs, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = inputs.shape
        segment_stride = segment_size // 2
        rest = segment_size - (segment_stride + seq_len %
                               segment_size) % segment_size

        if rest > 0:
            pad = self.zeros((batch_size, dim, rest), mindspore.float32)
            inputs = self.op2([inputs, pad])

        pad_aux = self.zeros((
            batch_size, dim, segment_stride), mindspore.float32)
        inputs = self.op2([pad_aux, inputs, pad_aux])
        return inputs, rest

    def create_chuncks(self, inputs, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)
        input0, rest = self.pad_segment(inputs, segment_size)
        batch_size, dim, _ = input0.shape
        segment_stride = segment_size // 2

        segments1 = input0[:, :, :-segment_stride].view(batch_size, dim, -1, segment_size)
        segments2 = input0[:, :, segment_stride:].view(batch_size, dim, -1, segment_size)

        segments = self.transpose(self.op3([segments1, segments2]).view(
            batch_size, dim, -1, segment_size), self.input_perm)
        return segments, rest

    def merge_chuncks(self, inputs, rest):
        # merge the split features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = inputs.shape
        segment_stride = segment_size // 2
        input0 = self.transpose(inputs, self.input_perm).view(batch_size,
                                                              dim, -1, segment_size*2)

        input1 = input0[:, :, :, :segment_size].view(
            batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input0[:, :, :, segment_size:].view(
            batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]
        return output

    def construct(self, inputs):
        # create chunks
        enc_segments, enc_rest = self.create_chuncks(
            inputs, self.segment_size)
        # separate
        output_all = self.rnn_model(enc_segments)

        # merge back audio files
        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_chuncks(output_all[ii], enc_rest)
            output_all_wav.append(output_ii)
        return output_all_wav


def capture_init(init):
    """
    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self.kwarg`
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self.kwarg = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__

class SWave(nn.Cell):
    @capture_init
    def __init__(self, N, L, H, R, C, sr, segment, input_normalize):
        super(SWave, self).__init__()
        # hyper-parameter
        self.N, self.L, self.H, self.R, self.C, self.sr, self.segment = N, L, H, R, C, sr, segment
        self.input_normalize = input_normalize
        self.context_len = 2 * self.sr / 1000
        self.context = int(self.sr * self.context_len / 1000)
        self.layer = self.R
        self.filter_dim = self.context * 2 + 1
        self.num_spk = self.C
        self.stack = ops.Stack()
        # similar to dprnn paper, setting chancksize to sqrt(2*L)
        self.segment_size = int(
            np.sqrt(2 * self.sr * self.segment / (self.L/2)))
        # model sub-networks`
        self.encoder = Encoder(L, N)
        self.decoder = Decoder(L)
        self.separator = Separator(self.filter_dim + self.N, self.N, self.H,
                                   self.filter_dim, self.num_spk, self.layer, self.segment_size, self.input_normalize)
        # init
        for p in self.get_parameters():
            if p.dim() > 1:
                initializer(HeNormal(), p.shape, mindspore.float32)

    def construct(self, mixture):
        mixture_w = self.encoder(mixture)
        output_all = self.separator(mixture_w)
        # fix time dimension, might change due to convolution operations
        T_mix = mixture.shape[-1]
        # generate wav after each RNN block and optimize the loss
        outputs = []
        for ii in range(len(output_all)):
            output_ii = output_all[ii].view(
                mixture.shape[0], self.C, self.N, mixture_w.shape[2])
            output_ii = self.decoder(output_ii)

            T_est = output_ii.shape[-1]
            output_ii = output_ii[:, :, 0:T_mix-T_est]
            outputs.append(output_ii)
        return self.stack(outputs)


class Encoder(nn.Cell):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.relu = ops.ReLU()
        self.expand_dims = ops.ExpandDims()
        # setting 50% overlap
        self.conv = nn.Conv1d(
            1, N, kernel_size=L, stride=L // 2, has_bias=False, pad_mode="pad", weight_init="HeNormal")

    def construct(self, mixture):
        mixture = self.expand_dims(mixture, 1)
        mixture_w = self.relu(self.conv(mixture))
        return mixture_w

def matrix():
    a = np.zeros(31996, dtype=np.int16)
    for i in range(1, 31996):
        if i % 4 == 0:
            a[i] = a[i-4]+1
        else:
            a[i] = a[i-1]+1
    mat = np.zeros((8002, 31996), dtype=np.int16)
    for i in range(31996):
        mat[a[i]][i] = 1
    mat = Tensor(mat, dtype=mindspore.float32)
    transpose = ops.Transpose()
    mat = transpose(mat, (1, 0))
    return mat


class Decoder(nn.Cell):
    def __init__(self, L):
        super(Decoder, self).__init__()
        self.L = L
        self.zeros = ops.Zeros()
        self.transpose = ops.Transpose()
        self.op2 = ops.Concat(2)
        self.mat = matrix()

    def gcd(self, a, b):
        if a < b:
            m = b
            n = a
        else:
            m = a
            n = b
        r = m % n
        while r != 0:
            m = n
            n = r
            r = m % n
        return n

    def overlap_and_add(self, signal, frame_step):
        """Reconstructs a signal from a framed representation.

        Adds potentially overlapping frames of a signal with shape
        `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
        The resulting tensor has shape `[..., output_size]` where

            output_size = (frames - 1) * frame_step + frame_length

        Args:
            signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
            frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

        Returns:
            A Tensor with shape [..., output_size] containing the overlap-added
            frames of signal's inner-most two dimensions.
            output_size = (frames - 1) * frame_step + frame_length

        Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
        """
        outer_dimensions = signal.shape[:-2]
        _, frame_length = signal.shape[-2:]

        # gcd=Greatest Common Divisor
        subframe_length = self.gcd(frame_length, frame_step)

        subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
        subframe_signal = self.transpose(subframe_signal, (0, 1, 3, 2))
        result = ops.matmul(subframe_signal, self.mat)
        result = self.transpose(result, (0, 1, 3, 2))
        result = result.view(*outer_dimensions, -1)
        return result

    def construct(self, est_source):
        est_source = self.transpose(est_source, (0, 1, 3, 2))
        pool = nn.AvgPool2d((1, self.L), stride=(1, self.L))
        est_source = pool(est_source)
        est_source = self.overlap_and_add(est_source, self.L//2)
        return est_source
