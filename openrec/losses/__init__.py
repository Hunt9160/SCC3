import copy
from torch import nn
from .ctc_loss import CTCLoss

support_dict = ['CTCLoss']


def build_loss(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'loss only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


class GTCLoss(nn.Module):

    def __init__(self,
                 gtc_loss,
                 gtc_weight=1.0,
                 ctc_weight=1.0,
                 zero_infinity=True,
                 **kwargs):
        super(GTCLoss, self).__init__()
        self.ctc_loss = CTCLoss(zero_infinity=zero_infinity)
        self.gtc_loss = build_loss(gtc_loss)
        self.gtc_weight = gtc_weight
        self.ctc_weight = ctc_weight

    def forward(self, predicts, batch):
        ctc_loss = self.ctc_loss(predicts['ctc_pred'],
                                 [None] + batch[-2:])['loss']
        gtc_loss = self.gtc_loss(predicts['gtc_pred'], batch[:-2])['loss']
        return {
            'loss': self.ctc_weight * ctc_loss + self.gtc_weight * gtc_loss,
            'ctc_loss': ctc_loss,
            'gtc_loss': gtc_loss
        }
