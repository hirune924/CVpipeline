#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.optim as optim

def get_optimizer_from_name(opt_name=None, model=None, **kwargs):

    if opt_name == 'sgd':
        optimizer = optim.SGD(params = model.parameters(),
                              **kwargs)
    else:
        print('{} is not implimented'.format(opt_name))
        optimizer = None

    return optimizer
