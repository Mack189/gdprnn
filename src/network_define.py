
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        net (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """
    def __init__(self, net, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss_fn
        self.cast = ops.Cast()
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()

    def construct(self, padded_mixture, mixture_lengths, padded_source):
        padded_mixture = padded_mixture.astype(mindspore.float32)
        padded_source = padded_source.astype(mindspore.float32)
        estimate_source = self._net(padded_mixture)
        estimate_source = estimate_source.astype(mindspore.float32)
        loss = 0
        cnt = len(estimate_source)
        for c_idx, est_src in enumerate(estimate_source):
            coeff = (c_idx+1)*(1.0/cnt)
            sisnr_loss, _, est_src, _ = self._loss(padded_source, est_src, mixture_lengths)
            loss += (coeff * sisnr_loss)
        loss /= len(estimate_source)
        return loss
