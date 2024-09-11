# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from atmg.config import LazyCall as L
from atmg.evaluation.retrieval import ZeroShotRetrievalEvaluator


evaluator = L(ZeroShotRetrievalEvaluator)(
    datasets=["coco", "flickr30k"],
    data_dir="datasets/eval",
    image_size=224,
)
