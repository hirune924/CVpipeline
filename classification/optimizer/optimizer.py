#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.optim as optim

def get_optimizer_from_name(opt_name=None, model=None, **kwargs):

    if opt_name == 'sgd':
        lr = kwargs['lr'] if 'lr' in kwargs else 0.1
        momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.0
        dampening = kwargs['dampening'] if 'dampening' in kwargs else 0.0
        weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0.0
        nesterov = kwargs['nesterov'] if 'nesterov' in kwargs else False

        optimizer = optim.SGD(params = model.parameters(),
                              lr = lr,
                              momentum = momentum,
                              dampening = dampening,
                              weight_decay = weight_decay,
                              nesterov = nesterov)
    else:
        print('{} is not implimented'.format(opt_name))
        optimizer = None

    return optimizer
