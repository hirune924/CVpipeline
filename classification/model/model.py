#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torchvision.models as models
import torch.nn as nn

def get_model_from_name(model_name=None, image_size=None, num_classes=None, pretrained=True):

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        print('{} is not implimented'.format(model_name))
        model = None

    return model
