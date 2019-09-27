#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.optim as optim

def get_optimizer_from_name(opt_name=None, model=None, target=[], opt_params={}, mode='yaml', lib='torch'):
    if lib=='torch':
        lib=optim

    if mode == 'yaml':
        if len(target) == 0:
            optimizer = getattr(lib, opt_name)(model.parameters(), **opt_params)
        else:
            target = make_params_list(model=model, target=target)
            optimizer = getattr(lib, opt_name)(target, **opt_params)

    elif mode == 'custom':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    return optimizer

def make_params_list(model=None, target=[]):
    for idx, d in enumerate(target):
        target[idx]['params'] = getattr(model, d['params']).parameters()

    return target
