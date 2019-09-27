#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.optim.lr_scheduler as torch_scheduler
import scheduler.custom_scheduler as custom_scheduler

def get_scheduler_from_name(scheduler_name=None, optimizer=None, scheduler_params={}, mode='yaml', lib='torch'):
    if lib=='torch':
        lib=torch_scheduler
    elif lib=='custom':
        lib = custom_scheduler

    if mode == 'yaml':
        scheduler = getattr(lib, scheduler_name)(optimizer, **scheduler_params)

    elif mode == 'custom':
        scheduler = torch_scheduler(optimizer, [5, 15], gamma=0.1)

    return scheduler
