# PyTorch, Torchvision
import torch
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from torchvision.io import write_video

# Common
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from IPython.display import Video

# Utils from Torchvision
tensor_to_image = ToPILImage()
image_to_tensor = ToTensor()

# Single sample video comparison
def make_comparison_grid(
    scene_rgb, scene_modal_mask,
    gt_amodal_rgb, gt_amodal_mask,
    pred_amodal_rgb, pred_amodal_mask,
    n_frames=24):
    """
    Makes a grid_tensor for a 3 row by 2 column video
    """

    grid_tensors = []
    for i in range(n_frames):
        # Select the i-th frame for each tensor
        scene_rgb_frame = scene_rgb[:, i]                 # (3, H, W)
        scene_mask_frame = scene_modal_mask[:, i]         # (1 or 3, H, W)
        gt_rgb_frame = gt_amodal_rgb[:, i]                # (3, H, W)
        gt_mask_frame = gt_amodal_mask[:, i]              # (1 or 3, H, W)
        pred_rgb_frame = pred_amodal_rgb[:, i]            # (3, H, W)
        pred_mask_frame = pred_amodal_mask[:, i]          # (1 or 3, H, W)

        # If mask is single-channel, repeat to 3 channels for visualization
        if scene_mask_frame.shape[0] == 1:
            scene_mask_frame = scene_mask_frame.repeat(3, 1, 1)
        if gt_mask_frame.shape[0] == 1:
            gt_mask_frame = gt_mask_frame.repeat(3, 1, 1)
        if pred_mask_frame.shape[0] == 1:
            pred_mask_frame = pred_mask_frame.repeat(3, 1, 1)

        # Top row: Scene RGB | Scene modal mask
        top_row = [scene_rgb_frame, scene_mask_frame]
        # Middle row: GT amodal RGB | GT amodal mask
        middle_row = [gt_rgb_frame, gt_mask_frame]
        # Bottom row: Pred amodal RGB | Pred amodal mask
        bottom_row = [pred_rgb_frame, pred_mask_frame]

        # Stack all rows horizontally (2 columns per row, 3 rows)
        grid = top_row + middle_row + bottom_row  # [6 tensors]
        grid_img = make_grid(grid, nrow=2, padding=2, pad_value=127)
        grid_tensors.append(grid_img)
    return grid_tensors

# Quick video
def make_vid(grid_tensors, save_path):
    vid_tensor = torch.stack(grid_tensors, dim=1).permute(1, 2, 3, 0)
    vid_tensor = (vid_tensor*255).long()
    write_video(save_path, vid_tensor, fps=5, options={'crf':'20'})


# Make matplotlib video with titles
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

def make_grid_video_with_titles(grid_tensors, 
                                panel_titles=[['Input RGB', 'Input Modal Mask'],
                                              ['True Amodal RGB', 'True Amodal Mask'],
                                              ['Pred Amodal RGB', 'Pred Amodal Mask']], 
                                nrows=3, ncols=2, figsize=(8, 12), interval=200, save_path=None):
    """
    Writes a 2-column, 3-row video. 

    grid_tensors: list of torch.Tensor, each of shape (3, H, W), values in [0,1] or [0,255]
    panel_titles: 2D list of strings, shape (nrows, ncols)
    interval: time between frames in ms
    save_path: if not None, saves animation to this path (e.g., 'output.mp4')
    """
    # Prepare figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes if isinstance(axes, np.ndarray) else np.array([[axes]])

    # Compute panel height and width from first frame
    img = grid_tensors[0].cpu()
    # Always convert to uint8 for display
    if img.dtype != torch.uint8:
        img = (img * 255).clamp(0, 255).byte()
    img_np = img.permute(1, 2, 0).numpy()  # (H, W, 3)

    # Debug: print min/max for first frame
    print(f"First frame min: {img_np.min()}, max: {img_np.max()}, dtype: {img_np.dtype}")

    H, W, _ = img_np.shape
    panel_h = H // nrows
    panel_w = W // ncols

    # Set titles and initialize images
    ims = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            ax = axes[i, j]
            ax.set_title(panel_titles[i][j])
            ax.axis('off')
            # Dummy initial image
            panel = img_np[i*panel_h:(i+1)*panel_h, j*panel_w:(j+1)*panel_w]
            im = ax.imshow(panel)
            row.append(im)
        ims.append(row)

    def update(frame_idx):
        img = grid_tensors[frame_idx].cpu()
        if img.dtype != torch.uint8:
            img = (img * 255).clamp(0, 255).byte()
        img_np = img.permute(1, 2, 0).numpy()
        for i in range(nrows):
            for j in range(ncols):
                y0, y1 = i * panel_h, (i + 1) * panel_h
                x0, x1 = j * panel_w, (j + 1) * panel_w
                panel = img_np[y0:y1, x0:x1]
                ims[i][j].set_data(panel)
        return sum(ims, [])

    ani = animation.FuncAnimation(fig, update, frames=len(grid_tensors), interval=interval, blit=False)
    plt.tight_layout()
    if save_path is not None:
        ani.save(save_path, writer='ffmpeg')
    else:
        plt.show()

