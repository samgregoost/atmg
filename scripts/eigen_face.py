# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate a trained model using implementations from `meru.evaluation` module.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torchvision import transforms as T

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from loguru import logger

from meru.config import LazyConfig, LazyFactory
from meru.utils.checkpointing import CheckpointManager
from meru.tokenizer import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--config", help="Path to an evaluation config file (.py)")
_AA("--checkpoint-path", help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")


def visualize_embeddings(dataloader_eigen, dataloader_face, model):
    img_list = []
    text_list = []
    dataloader_iter_eigen = iter(dataloader_eigen)
    dataloader_iter_face = iter(dataloader_face)
   # for iteration in range(1, 100):

    batch_eigen, _ = next(dataloader_iter_eigen)
    batch_face, _ = next(dataloader_iter_face)



    with torch.inference_mode():
        eigen_feats = model.encode_image(batch_eigen.to(model.device), project=True)
        face_feats = model.encode_image(batch_face.to(model.device), project=True)
      #  text_feats = model.encode_text(tokens, project=True)
        img_list.append(torch.norm(eigen_feats, dim=-1))
        text_list.append(torch.norm(face_feats, dim=-1))

    image_vecs = torch.stack(img_list, dim=0).reshape(-1).detach().cpu().numpy()
    text_vecs = torch.stack(text_list, dim=0).reshape(-1).detach().cpu().numpy()

    # Plotting the histogram again
#    with plt.rc('font', size=16):
    plt.figure(figsize=(8, 6))
    plt.hist(image_vecs, bins=30, color='skyblue', edgecolor='black', label='Eigen faces')
    plt.hist(text_vecs, bins=30, color='red', edgecolor='black', label = 'Face images')
 #   plt.title("S")
 #3   plt.xticks(fontsize=14)                  # Set font size for X-axis tick labels
   # plt.yticks(fontsize=14) 
    plt.legend(fontsize=18)
    plt.xlabel("Distance", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)

    # Saving the histogram
    file_path = "./eigen_histogram.png"
    plt.savefig(file_path, dpi=300)

def main(_A: argparse.Namespace):
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)
 #   print(_A.config)
#    _C = LazyConfig.load(_A.config)
  #  logger.info(OmegaConf.to_yaml(_C))

   # logger.info("Command line args:")
   # for arg in vars(_A):
   #     logger.info(f"{arg:<20}: {getattr(_A, arg)}")

   # logger.info(f"Evaluating checkpoint in {_A.checkpoint_path}...")

    # Create a fresh model and evaluator for every checkpoint, so the evaluator
    # is free to modify the model weights (e.g. remove projection layers).
    image_eigen_path = '/home/ubuntu/ebs/CelebAMask-HQ/CelebA-HQ-img'
    image_face_path = '/home/ubuntu/ebs/eigenfaces'

    # Define any transformations you want to apply to the images
    image_transform = T.Compose(
        [T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]
    )

    # Create a Dataset from the image folder
    dataset_eigen = ImageFolder(root=image_eigen_path, transform=image_transform)
    dataset_face = ImageFolder(root=image_face_path, transform=image_transform)

    # Create a DataLoader
    dataloader_eigen = DataLoader(dataset_eigen, batch_size=1000, shuffle=True, num_workers=4)
    dataloader_face = DataLoader(dataset_face, batch_size=1000, shuffle=True, num_workers=4)


    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)

   # dataloader = LazyFactory.build_dataloader(_C_TRAIN)
    visualize_embeddings(dataloader_eigen, dataloader_face , model)
    # results_dict = evaluator(model)
    #
    # # Log results for copy-pasting to spreadsheet, including checkpoint path.
    # header = ",".join(results_dict.keys())
    # numbers = ",".join([f"{num:.1f}" for num in results_dict.values()])
    #
    # logger.info(f"copypaste: {_A.checkpoint_path}")
    # logger.info(f"\ncopypaste below:\n{header}\n{numbers}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)

