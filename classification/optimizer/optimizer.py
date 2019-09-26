#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.optim as optim

def get_optimizer_from_name(opt_name=None, model=None, opt_params={}):

    optimizer = getattr(optim, opt_name)(params=model.parameters(), **opt_params)

    return optimizer
