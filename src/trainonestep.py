
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel._auto_parallel_context import auto_parallel_context

class TrainOneStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0, use_global_norm=True, clip_global_norm_value=5.0):
        super(TrainOneStep, self).__init__(network, optimizer, sens)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = float(sens)
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)


    def construct(self, padded_mixture, mixture_lengths, padded_source):
        loss = self.network(padded_mixture, mixture_lengths, padded_source)
        sens = P.Fill()(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(padded_mixture, mixture_lengths, padded_source, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if self.use_global_norm:
            grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss
