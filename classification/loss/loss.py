#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

def get_loss_from_name(loss_name=None, **kwargs):

    if loss_name == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss(**kwargs)
    else:
        print('{} is not implimented'.format(loss_name))
        loss = None

    return loss
