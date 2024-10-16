# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .train_atmg_vit_l import dataset, model, optim, train


model.visual.arch = "vit_base_patch16_224"
