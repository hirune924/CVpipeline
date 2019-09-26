#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.optim as optim

def get_optimizer_from_name(opt_name=None, model=None, target=[], opt_params={}):
    target = make_params_list(model=model, target=target)
    print(target)
    optimizer = getattr(optim, opt_name)(target, **opt_params)

    return optimizer

def make_params_list(model=None, target=[]):
    if len(target) == 0:
        target = {'params': model.parameters()}
    else:
        for idx, d in enumerate(target):
            target[idx]['params'] = getattr(model, d['params']).parameters()

    return target
