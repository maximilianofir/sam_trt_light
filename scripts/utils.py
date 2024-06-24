#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# li
# mitations under the License.
#
import os

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = anns #sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.55]])
        img[m] = color_mask
    ax.imshow(img)

def create_dir(dir):
    if not os.path.exists(dir):
        print(f"Directory {dir} does not exist. Creating..")
        os.makedirs(dir)

def coords(s):
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.TypeError("Coordinates must be x,y")

def resize_longest_image_size(
        input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

def visualize_point_and_mask(path, image, mask, point, labels, slice_idx):
    mask_slice = slice(slice_idx, slice_idx+1)
    mask = mask[:, mask_slice, :, :]
    aspect = plt.figaspect(image)
    fig, _ = plt.subplots(figsize=(aspect[0]*2, aspect[1]*2))
    fig.subplots_adjust(0,0,1,1)
    plt.imshow(image)
    show_mask(mask.cpu().numpy(), plt.gca())
    show_points(point, labels, plt.gca())
    plt.axis('off')
    plt.savefig(path)

def visualize_multiple_masks(path, image, masks):
    aspect = plt.figaspect(image)
    fig, _ = plt.subplots(figsize=(aspect[0]*2, aspect[1]*2))
    fig.subplots_adjust(0,0,1,1)
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(path)
