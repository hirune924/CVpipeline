#!/usr/bin/env python
# -*- coding: utf-8 -*-
import metric.metric_lib as metric_lib
        
def get_metrics(metric_options=[]):

    metric_dict = {}
    for m in metric_options:
        metric_dict[m['metric_name']] = getattr(metric_lib, m['metric_name'])(**m['metric_params'])

    return metric_dict

