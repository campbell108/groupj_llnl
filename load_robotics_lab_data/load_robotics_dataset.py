#!/usr/bin/venv_dssi python
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from PIL import Image
import glob
import re

'''
Next steps: 
    1. Add a download of the data function
    2. Add my resize transformation (256, 256) function from the condensed tar frames
    3. [optional] Unique masks add on + param for selecting mask files
'''

class FrameMaskVideoDataset(Dataset):
    def __init__(self, frames_dir, masks_dir, clip_len=8, transform=None, mask_prefix=None):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.clip_len = clip_len
        self.transform = transform or ToTensor()
        self.mask_prefix = mask_prefix
        self.samples = self._build_index() # num of clips found

    def _build_index(self): 
        frame_files = sorted(glob.glob(os.path.join(self.frames_dir, "*.jpg")))
        samples = []

        for i in range(0, len(frame_files) - self.clip_len + 1):
            clip_frames = frame_files[i:i + self.clip_len]

            # Extract indices from filenames like frame_0001.jpg → 0001
            frame_indices = [
                re.search(r"(\d+)", os.path.basename(fp)).group(1) for fp in clip_frames
            ]

            # For each index, find all matching mask files (e.g., mask1_0001.pt, mask2_0001.pt, ...)
            clip_mask_groups = []
            for idx in frame_indices:
                matching_masks = sorted(glob.glob(os.path.join(self.masks_dir, f"*_{idx}.pt")))
                if not matching_masks:
                    break  # skip this clip if *any* frame is missing masks
                clip_mask_groups.append(matching_masks)

            if len(clip_mask_groups) == self.clip_len:
                samples.append((clip_frames, clip_mask_groups))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, all_mask_paths = self.samples[idx]

        # Load frames
        frames = [self.transform(Image.open(fp)) for fp in frame_paths]
        frames_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]

        # Load all masks for each frame
        # Output shape: [T, N, H, W]  (T=frames, N=masks per frame)
        all_masks = []
        for mask_group in all_mask_paths:
            masks = [torch.load(mp).squeeze(0).float() for mp in mask_group]
            stacked = torch.stack(masks, dim=0)  # [N, H, W]
            all_masks.append(stacked)

        masks_tensor = torch.stack(all_masks, dim=0)  # [T, N, H, W]

        return {
            'frame': frames_tensor,     # [3, T, H, W]
            'modal_mask': masks_tensor       # [T, N, H, W]
        }

'''
dataset = FrameMaskVideoDataset(
    frames_dir="./frames_resized/CVTrialRun_1_VialsNoColor_v1",
    masks_dir="../data/masks_resized",
    clip_len=10,
    mask_prefix=None
)

sample = dataset[0]
print(sample['frame'].shape)  # torch.Size([3, 10, H, W]) ([RGB, n_frames in the clip, height, width])
print(sample['modal_mask'].shape)   # torch.Size([10, M/F, H, W]) ([n_frames, masks/frame, height, width])
# 6 masks per frame
'''