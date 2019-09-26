#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

def get_loss_from_name(loss_name=None, loss_params={}):

    loss = getattr(nn, loss_name)(**loss_params)

    return loss
