#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

def get_loss_from_name(loss_name=None, **kwargs):

    if loss_name == 'CrossEntropyLoss':
        weight = kwargs['weight'] if 'weight' in kwargs else None
        size_average = kwargs['size_average'] if 'size_average' in kwargs else None
        ignore_index = kwargs['ignore_index'] if 'ignore_index' in kwargs else -100
        reduce = kwargs['reduce'] if 'reduce' in kwargs else None
        reduction = kwargs['reduction'] if 'reduction' in kwargs else 'mean'


        loss = nn.CrossEntropyLoss(weight = weight, 
                                   size_average = size_average, 
                                   ignore_index = ignore_index, 
                                   reduce = reduce, 
                                   reduction = reduction)
    else:
        print('{} is not implimented'.format(loss_name))
        loss = None

    return loss
