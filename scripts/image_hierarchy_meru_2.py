# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path


import pickle
import torch
import torchvision.transforms as T
from loguru import logger
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from meru import lorentz as L
from meru.evaluation.catalog import DatasetCatalog
from meru.evaluation.class_names import CLASS_NAMES
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
import random
from torchvision.utils import save_image

import argparse
import json

import torch
from PIL import Image
from torchvision import transforms as T

from meru import lorentz as L
from meru.config import LazyConfig, LazyFactory
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
from meru.utils.checkpointing import CheckpointManager
from meru import lorentz as L
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--checkpoint-path", help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--image-path", help="Path to an image (.jpg) for perfoming traversal.")
_AA("--steps", type=int, default=50, help="Number of traversal steps.")


image_path = '/home/ubuntu/ebs/meru/datasets/eval/imagenet/train'

from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder"""
    
    # Override the __getitem__ method. This is the method that dataloader calls
    def __getitem__(self, index):
        # This is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    """
    Interpolate between given feature vector and `[ROOT]` depending on model type.
    """

    # Linear interpolation between root and image features. For MERU, this happens
    # in the tangent space of the origin.
    if isinstance(model, MERU):
        feats = L.log_map0(feats, model.curv.exp())

    interp_feats = [
            torch.lerp(root_feat, feats, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)

    # Lift on the Hyperboloid (for MERU), or L2 normalize (for CLIP).
    if isinstance(model, MERU):
        feats = L.log_map0(feats, model.curv.exp())
        interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    else:
        interp_feats = torch.nn.functional.normalize(interp_feats, dim=-1)

    # Reverse the traversal order: (image first, root last)
    return interp_feats.flip(0)

def calc_scores(
    model, image_feats: torch.Tensor, text_feats: torch.Tensor, has_root: bool
):
    """
    Calculate similarity scores between the given image and text features depending
    on model type.

    Args:
        has_root: Flag to indicate whether the last text embedding (at dim=0)
            is the `[ROOT]` embedding.
    """

    if isinstance(model, MERU):
        scores = -L.pairwise_dist(image_feats, text_feats, model.curv.exp())

   #     scores_ = L.pairwise_dist(image_feats, text_feats, model.curv.exp())

 #       scores_ = torch.abs(torch.acosh(torch.clamp(1+torch.norm(image_feats,dim=-1), min=1 + 10e-10))[:, None] - torch.acosh(torch.clamp(1+torch.norm(text_feats,dim=-1), min=1 + 10e-10))[None, :])
    #    print(scores_)
     #   scores = (L.pairwise_oxy_angle(image_feats, text_feats, model.curv.exp()) - 100*scores_)        
        # For MERU, exclude text embeddings that do not entail the given image.
        # _aper = L.half_aperture(text_feats, model.curv.exp())
        # _oxy_angle = L.oxy_angle(
        #     text_feats[:, None, :], image_feats[None, :, :], model.curv.exp()
        # )
        # entailment_energy = _oxy_angle - _aper[..., None]
        #
        # # Root entails everything.
        # if has_root:
        #     entailment_energy[-1, ...] = 0
        #
        # # Set a large negative score if text does not entail image.
        # scores[entailment_energy.T > 0] = -1e12
        
        print(scores)
        print(scores.shape)
        return scores
    else:
        # model is not needed here.
        return image_feats @ text_feats.T


def get_image_feat(model, image_feats, text_feat):
    scores = -L.pairwise_dist(image_feats, text_feat, model.curv.exp())
    print(scores.shape)
    return scores


@torch.inference_mode()
def main(_A: argparse.Namespace):
    # Get the current device (this will be `cuda:0` here by default) or use CPU.
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create the model using training config and load pre-trained weights.
    _C_TRAIN = LazyConfig.load('configs/train_meru_vit_s.py')
    model = LazyFactory.build_model(_C_TRAIN, device).eval()

    CheckpointManager(model=model).load(_A.checkpoint_path)

    image_transform = T.Compose(
        [
            T.Resize(224, T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )

#    loader = DataLoader(
 #       DatasetCatalog.build(
 #           'imagenet', 'datasets/eval', "train", image_transform
 #       ),
 #       batch_size=128,
 #       num_workers=0,
 #   )
    
    dataset = ImageFolderWithPaths(root=image_path, transform=image_transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

  #  image_feats, labels, all_images = _encode_dataset(loader, model, project=True)
   # image_feats_e, labels, all_images = _encode_dataset(loader, model, project=False)
    # Features returned by this function will be on CPU, move to device:
    
    with open('img_hierarchy_features.pkl', 'rb') as file:
        image_feats = pickle.load(file)

    with open('img_paths.pkl', 'rb') as file:
        img_paths = pickle.load(file)
    
    image_feats = torch.cat(image_feats, dim=0)
    all_images = [item for sublist in img_paths for item in sublist]
    print(image_feats.shape)
    print(len(img_paths))
    image_feats = image_feats.to(model.device)

 #   image_index = random.randint(0, image_feats.shape[0])
#    image_feat_e = model.encode_image( all_images[image_index][None, ...].to(model.device), False)

    tokenizer = Tokenizer()
    prompt = "A photo of a man playing a guitar"
    caption_tokens = tokenizer([prompt])
    text_feat = model.encode_text(caption_tokens, project=True)

 #   print(f"image_index {image_index}")
    scores = get_image_feat(model, image_feats, text_feat).T
    _, returned_image_idx = scores.max(dim=-1)
    returned_image_path = all_images[returned_image_idx.item()]
    returned_image = Image.open(returned_image_path)

    # Save the image with a new name or format
    returned_image.save('returned_image.png')
#    save_image(returned_image, f'returned_image.png')
  #  interp_feat_list = []
 #   for i in range(40):
  #      current_image_feat =  image_feat_e * (1 + 0.01*i)
  #      interp_feat_list.append(current_image_feat)

  #  interp_feats = torch.cat(interp_feat_list, dim=0)

   # interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    

    root_feat = torch.zeros(_C_TRAIN.model.embed_dim, device=device)
 #   interp_feats = interpolate(model, image_feats[returned_image_idx][None, ...], root_feat, 40)
    interp_feats = interpolate(model, text_feat[0], root_feat, 40)
 #   nn1_scores = calc_scores(model, interp_feats, image_feats, has_root=True)
    print(interp_feats.shape)
    print(image_feats.shape)
    nn1_scores = get_image_feat(model, image_feats, text_feat).T
 #   nn1_scores = get_image_feat(model, image_feats, interp_feats).T
    print(nn1_scores.shape)
#    nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
    nn1_scores, _nn1_idxs = torch.topk(nn1_scores, 40, dim=-1)
    print(_nn1_idxs.shape)
    nn1_features = [image_feats[_idx.item()] for _idx in _nn1_idxs[0]]
    nn1_features = torch.stack(nn1_features)
    print(nn1_features.shape)
        
    norms = nn1_features.norm(p=2, dim=1)
    sorted_indices = norms.argsort().tolist()
    nn1_images = [all_images[_idx.item()] for _idx in _nn1_idxs[0]]
    nn1_images = [nn1_images[i] for i in sorted_indices]
    norms = [norms[i] for i in sorted_indices]
    i = 0
    for image in nn1_images:
        print(norms[i])
        image = Image.open(image)
        image.save(f'output_image_{i}.png')
        i = i+1
      #  image_tensor = image.cpu()
      #  image_tensor = image_tensor.detach()
      #  save_image(image_tensor, f'output_image_{i}.png')
     #   i = i + 1



def _encode_dataset(
    data_loader: DataLoader,
    model: MERU | CLIPBaseline,
    project: bool,
):
    """
    Extract image features and labels for a given dataset using the given model.

    Args:
        data_loader: PyTorch dataset or dataloader that serves instances/batches
            of `(image, label)` tuples.
        model: Model that implements `encode_image` method to extract features.
        project: Input argument to `model.encode_image`.
    """

    # Collect batches of extracted image features and labels (as-is from loader).
    all_image_feats, all_labels, all_images = [], [], []

    i = 0
    for images, labels, paths in tqdm(data_loader, desc=f"Extracting image feats"):
      #  print(i)
     #   if i > 194:
       #     print("test")
    #    print(paths)
        with torch.inference_mode():
            image_feats = model.encode_image(images.to(model.device), project)


        all_image_feats.append(image_feats.cpu())
 #       all_labels.append(labels)
        all_images.append(paths)

    with open('img_hierarchy_features.pkl', 'wb') as file:
        pickle.dump(all_image_feats, file)
    with open('img_paths.pkl', 'wb') as file:
        pickle.dump(all_images, file)
       # i = i + 1
    return torch.cat(all_image_feats, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_images, dim=0)



if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
