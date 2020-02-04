import torch
from torch.nn import Parameter
from functools import wraps

"""
Code from https://github.com/salesforce/awd-lstm-lm
paper: https://arxiv.org/pdf/1708.02182.pdf 
for a difference between dropout and drop connect see https://stats.stackexchange.com/questions/201569/what-is-the-difference-between-dropout-and-drop-connect?fbclid=IwAR26B772sQXXBBpNSI_cu41zIEFu0gVkKgn34E1Wbbb4EbQdkdvmMOgro4s
"""


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        """

        :param module: nn.Linear or nn.LSTM
        :param weights: which weights to apply dropout (names of weights of module)
        :param dropout: dropout to be applied
        :param variational: if True applies variational dropout, if False applies drop connect (different masks!!!)
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        """
        this function renames each 'initial weight name' to 'initial weight name' + '_raw'
        :return:
        """
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


if __name__ == '__main__':
    import torch
    from weight_drop import WeightDrop

    # Input is (seq, batch, input)
    # x = torch.autograd.Variable(torch.randn(2, 1, 10)).cuda()
    x = torch.autograd.Variable(torch.randn(2, 1, 10))
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = WeightDrop(torch.nn.Linear(10, 10), ['weight'], dropout=0.2, variational=True)
    # lin = WeightDrop(torch.nn.Linear(10, 10), ['weight'], dropout=0.2, variational=False)
    # lin.cuda()
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    ###

    print('Testing WeightDrop with LSTM')

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.9)
    # wdrnn.cuda()

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

    print('---')