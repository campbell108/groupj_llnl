#!/usr/bin/env python3
import torch
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from torchvision.io import write_video

# Common
from pathlib import Path
from PIL import Image
import numpy as np
import random
import json



def prepare_single_sample_for_plotting(sample):
    """
    Prepare a single sample for plotting in matplotlib
    Object-wise ModalMask+RGB -> Object-wise AmodalMask+AmodalContent

    Args:
        sample (dict): Single dictionary from dataset
    
    Returns:


    """
    # Prep ground truth data
    # True depth
    true_dep = sample['depths'][0][0]

    # True sample RGB
    # Currently c, n_frames, h, w
    # Remove time dimension and reorder into (H, W, C)
    true_rgb_content = sample['frames'].squeeze(1).permute(1,2,0)

    # Modal mask
    # currently c, n_frames, h, w
    # Remove time dimension and reorder into (H, W, C)
    true_modal_mask = sample['modal_masks'].squeeze(1).permute(1,2,0)

    # Amodal mask
    # currently c, n_frames, h, w
    # Remove time dimension and reorder into (H, W, C)
    true_amodal_mask = sample['amodal_masks'].squeeze(1).permute(1,2,0)

    # Amodal content
    # currently c, n_frames, h, w
    # Remove time dimension and reorder into (H, W, C)
    true_amodal_content = sample['amodal_content'].squeeze(1).permute(1,2,0)
    true_amodal_content = (true_amodal_content * 255).to(torch.uint8)

    return true_dep, true_rgb_content, true_modal_mask, true_amodal_mask, true_amodal_content

def get_single_unet_image_input(sample):
    """
    Prepare single sample for Unet_Image inference

    Args:
        sample (dict): Single sample from the dataset

    Returns:
        model_input (torch.tensor): Tensor input for inference. Needs to be shape [bs, c, h, w]

    """
    # Run the model
    print("Frames + MMask Shape")
    print(sample['frames'].shape, sample['modal_masks'].shape)


    # Concatenate along batch dimension
    # cat frames and modal_masks along the batch dimension - 1st element (channels)
    # Second element (ix=1) is n_frames - we don't need that (time dimension)
    model_input = torch.cat([sample['frames'], sample['modal_masks']], dim=0)  # [4, 1, 256, 256]
    print("after cat", model_input.shape)

    time_dim = 1 # ix=1 (n_frames)
    # time dimension for a sample is different to a batch because a sample doesn't have batch size!
    # Remove time dimension from a few things (add it back in later when you make your video model!)
    model_input = model_input.squeeze(time_dim) # remove time dimension (you will probably want it later!)
    print("Model input after squeezing n_frames dimension", model_input.shape)

    # Reshape to [1, 4, 256, 256]
    # Unsqueeze at zero dim - adds dimension
    model_input = model_input.unsqueeze(0)  # [1, 4, 256, 256]
    print("After unsqueeze at zero dim", model_input.shape)

    # 3. Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    model_input = model_input.to(device)
    print("Final model input", model_input.shape)
    return model_input

def get_single_unet_image_output_to_numpy(logits_amodal_content, logits_amodal_mask, true_amodal_content, true_amodal_mask):
    """
    Convert single sample Unet evaluation (logits), to plotting ready numpy arrays
    Unet_Image ground truth: true_amodal_content, true_amodal_mask

    Args:
        logits_amodal_content: torch.tensor, [bs, c, h, w]
        logits_amodal_mask: torch.tensor [bs, 1, c, h, w]
        true_amodal_content: numpy array [H, W, C]
        true_amodal_mask: numpy array [H, W, C]

    Returns:
        preds_amodal_content: numpy array [H, W, C]
        preds_amodal_mask: numpy array [H, W, C]
    """
    # 1. RGB Content
    # print("AMODAL RGB CONTENT")
    # print("Logits amodal content, initial (after masking): logits amodal content shape", logits_amodal_content.shape, 
    #     "range", logits_amodal_content.min(), logits_amodal_content.max())

    # These are masked amodal contents for the object
    # Apply sigmoid to logits for amodal content (no rounding!)
    logits_amodal_content = logits_amodal_content.sigmoid()
    # print("Logits amodal content, masked and sigmoid applied: logits amodal content shape", logits_amodal_content.shape, 
    #     "range", logits_amodal_content.min(), logits_amodal_content.max())

    # Squeeze to remove the batch dimensions
    preds_amodal_content = logits_amodal_content.squeeze(0)                   # [3, H, W]
    # print("Preds Amodal RGB, after squeeze: preds_amodal_content shape", 
    #     preds_amodal_content.shape, "range", preds_amodal_content.min(), preds_amodal_content.max())


    # Permute to send to matplotlib (H, W, C)
    preds_amodal_content = preds_amodal_content.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    # print("Preds Amodal RGB, after permute and numpy: preds_amodal_content shape", 
    #     preds_amodal_content.shape, "range", preds_amodal_content.min(), preds_amodal_content.max())

    # Apply clip
    preds_amodal_content = np.clip(preds_amodal_content, 0, 1)

    # Multiply by 255 to get color range
    preds_amodal_content = (preds_amodal_content * 255).astype(np.uint8)     # [0, 255], uint8
    # print("Preds Amodal RGB, after multiply and int: preds_amodal_content shape", 
    #     preds_amodal_content.shape, "range", preds_amodal_content.min(), preds_amodal_content.max())

    # print(f"AMODAL CONTENT: true {true_amodal_content.shape}, pred {preds_amodal_content.shape}")
    # print("range comparison\n", 
    #     f"true {true_amodal_content.min()}, {true_amodal_content.max()}\n",
    #     f"preds {preds_amodal_content.min()}, {preds_amodal_content.max()}\n")

    # check shape
    assert preds_amodal_content.shape == true_amodal_content.shape


    # 2. Amodal Mask
    # print("AMODAL RGB MASK")

    # print("Logits amodal mask, initial: shape", logits_amodal_mask.shape, 
    #     "range", logits_amodal_mask.min(), logits_amodal_mask.max())

    # Apply rounded sigmoids to get integers, binary
    preds_amodal_mask = logits_amodal_mask.sigmoid().round().to(torch.uint8)
    # This yields bs, c, h, w
    # print("Preds amodal mask, after sigmoid: shape", preds_amodal_mask.shape, 
    #     "range", preds_amodal_mask.min(), preds_amodal_mask.max())

    # Drop ix = 0 (batch dimension)
    preds_amodal_mask = preds_amodal_mask.squeeze(0)            # [1, H, W]
    # print("Preds amodal mask, after dropping bs: shape", preds_amodal_mask.shape, 
    #     "range", preds_amodal_mask.min(), preds_amodal_mask.max())


    # permute to send to matplotlib (H, W, C) and send to numpy
    preds_amodal_mask = preds_amodal_mask.permute(1, 2, 0).cpu().numpy()                 # For probability map (grayscale)
    # print("Preds amodal mask, after permute and send to numpy: shape", preds_amodal_mask.shape, 
    #     "range", preds_amodal_mask.min(), preds_amodal_mask.max())


    # print(f"AMODAL MASK: true {true_amodal_mask.shape}, pred {preds_amodal_mask.shape}")
    # print("range comparison\n", 
    #     f"true {true_amodal_mask.min()}, {true_amodal_mask.max()}\n",
    #     f"preds {preds_amodal_mask.min()}, {preds_amodal_mask.max()}\n")

    # check shape
    assert preds_amodal_content.shape == true_amodal_content.shape


    return preds_amodal_content, preds_amodal_mask