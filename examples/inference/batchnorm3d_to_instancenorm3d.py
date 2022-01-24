#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from copy import deepcopy

def batchnorm3d_to_instancenorm3d(model):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], num_converted = batchnorm3d_to_instancenorm3d(module)
            conversion_count += num_converted

        if isinstance(module, nn.BatchNorm3d):
            layer_new = nn.InstanceNorm3d(module.num_features, eps=module.eps, 
                                       momentum=module.momentum,
                                       affine=module.affine,
                                       track_running_stats=False)
            layer_new.weight=module.weight
            layer_new.bias = module.bias
            model._modules[name] = layer_new
            conversion_count += 1

    return model, conversion_count

if __name__ == '__main__':
    model = torch.identity()
    model2 = deepcopy(model).cpu()
    new_model, count = batchnorm3d_to_instancenorm3d(model2, nn.BatchNorm3d, nn.InstanceNorm3d)
