#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import loss.custom_loss as custom_loss 

def get_loss_from_name(loss_name=None, loss_params={}, mode='yaml', lib='torch'):
    if lib=='torch':
        lib = nn
    elif lib=='custom':
        lib = custom_loss

    if mode == 'yaml':
        loss = getattr(lib, loss_name)(**loss_params)
    elif mode == 'custom':
        loss = nn.CrossEntropyLoss()

    return loss
