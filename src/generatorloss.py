
import mindspore.nn  as nn
from src.network_define import WithLossCell
from src.models.loss import myloss

class Generatorloss(nn.Cell):
    def __init__(self, generator):
        super(Generatorloss, self).__init__()
        self.generator = generator
        self.my_loss = myloss()
        self.net_with_loss = WithLossCell(self.generator, self.my_loss)

    def construct(self, maxture, lens, source):
        loss = self.net_with_loss(maxture, lens, source)
        return loss
