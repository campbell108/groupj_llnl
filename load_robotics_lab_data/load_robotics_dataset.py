#!/usr/bin/venv_dssi python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import os
from PIL import Image
import numpy as np
import tarfile
import glob
import requests
import cv2
import re
import os

class FrameMaskVideoDataset(Dataset):
    def __init__(self, frames_dir, masks_dir, clip_len=8, transform=None, mask_prefix=None):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.clip_len = clip_len
        self.transform = transform or ToTensor()
        self.mask_prefix = mask_prefix
        self.samples = self._build_index() # num of clips found

    def _build_index(self): 
        frame_files = sorted(glob.glob(os.path.join(self.frames_dir, '*.jpg')))
        samples = []

        for i in range(0, len(frame_files) - self.clip_len + 1):
            clip_frames = frame_files[i:i + self.clip_len]

            # Extract indices from filenames like frame_0001.jpg → 0001
            frame_indices = [
                re.search(r'(\d+)', os.path.basename(fp)).group(1) for fp in clip_frames
            ]

            # For each index, find all matching mask files (e.g., mask1_0001.pt, mask2_0001.pt, ...)
            clip_mask_groups = []
            for idx in frame_indices:
                matching_masks = sorted(glob.glob(os.path.join(self.masks_dir, f'mask*_{idx}.pt')))
                if not matching_masks:
                    break  # skip this clip if any frame is missing masks
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
            'frames': frames_tensor,     # [3, T, H, W]
            'modal_masks': masks_tensor       # [T, N, H, W]
        }

def preprocess_transfer_test_frames(input_dir='./data/VialsWithColor_Every10th_Subset.tar.gz', output_dir='./data/frames', resizeFrames=True, output_size=(256, 256)): 
    os.makedirs(output_dir, exist_ok=True)
    with tarfile.open(input_dir, 'r:*') as tar:
        tar.extractall(path=output_dir)

    print(f'Tar File Extracted to: {output_dir}')

    if resizeFrames:
        base_name = os.path.basename(input_dir)

        if base_name.endswith('.tar.gz'):
            folder_name = base_name[:-7]
        elif base_name.endswith('.tgz'):
            folder_name = base_name[:-4]
        elif base_name.endswith('.tar'):
            folder_name = base_name[:-4]
        else:
            folder_name = os.path.splitext(base_name)[0]

        extracted_dir = f'{output_dir}/{folder_name}'
        extracted_resized_dir = f'{extracted_dir}/resized'
        os.makedirs(extracted_resized_dir, exist_ok=True)

        frame_files = sorted(glob.glob(os.path.join(extracted_dir, '*.jpg')))  
        for idx, frame_file in enumerate(frame_files):
            print(frame_file)
            frame = cv2.imread(frame_file)
            if frame is None:
                print(f'Warning: Could not read {frame_file}')
                continue
            frame_resized = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
            out_path = os.path.join(extracted_resized_dir, f'{idx:04d}.jpg')
            cv2.imwrite(out_path, frame_resized)

def preprocess_transfer_test_mask_tensors(input_dir='./data', output_dir='./data/VialsWithColor_Every10th_Subset/masks', resizeMasks=True, size=(256, 256)):
    '''
    The VialsWithColor_Every10th_Subset are the only masks available
    '''
    
    os.makedirs(output_dir, exist_ok=True)
    mask_files = sorted(glob.glob(os.path.join(input_dir, '*.pt')))
    
    for idx, mask_file in enumerate(mask_files):
        print(f'Current Mask file: {mask_file}')
        counter = 0 
        mask_batch = torch.load(mask_file)  # shape: (N, 1, H, W)
        
        if mask_batch.ndim != 4:
            raise ValueError(f"Expected shape (N, 1, H, W), got {mask_batch.shape}")
        
        for i in range(mask_batch.shape[0]):
            mask_tensor = mask_batch[i]  # shape: (1, H, W)
            mask_np = mask_tensor.squeeze(0).cpu().numpy()  # shape: (H, W)

            if mask_np.dtype == np.bool_:
                mask_np = mask_np.astype(np.uint8)

            if mask_np.size == 0:
                print(f"Skipping empty mask in file {mask_file}, index {i}")
                continue

            # Resize using nearest neighbor
            mask_resized = cv2.resize(mask_np, size, interpolation=cv2.INTER_NEAREST)

            # Convert back to tensor and add channel
            mask_resized_tensor = torch.from_numpy(mask_resized).unsqueeze(0)  # shape: (1, H, W)

            out_path = os.path.join(output_dir, f"mask{idx}_{counter:04d}.pt")
            torch.save(mask_resized_tensor, out_path)
            counter += 1

def get_transfer_test_frames(output_dir='./data'):
    os.makedirs(output_dir, exist_ok=True)
    url = 'https://huggingface.co/datasets/Amar-S/LLNL_DSC_2025/resolve/main/Robotics_Lab_Data/transfer_test_video/VialsWithColor_Every10th_Subset.tar.gz'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(f"{output_dir}/VialsWithColor_Every10th_Subset.tar.gz", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print('Transfer Test Frame Tar File Download Completed!')
    else:
        print("Failed to download Tar File:", response.status_code)\

def get_transfer_test_masks(output_dir='./data'):
    os.makedirs(output_dir, exist_ok=True)
    base_url = 'https://huggingface.co/datasets/Amar-S/LLNL_DSC_2025/resolve/main/Robotics_Lab_Data/transfer_test_video/'
    filenames = [
    'mask0_vialswithcolor_every10th_subset.pt',
    'mask1_vialswithcolor_every10th_subset.pt',
    'mask2_vialswithcolor_every10th_subset.pt',
    'mask3_vialswithcolor_every10th_subset.pt',
    'mask4_vialswithcolor_every10th_subset.pt',
    'mask5_vialswithcolor_every10th_subset.pt',
    ]

    for fname in filenames:
        url = base_url + fname
        print(f'Downloading {fname}...')
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'{output_dir}/{fname}', 'wb') as f:
                f.write(response.content)
            print(f'{fname} downloaded.')
        else:
            print(f'Failed to download {fname}, status code: {response.status_code}')
    print('Transfer Test Mask Files Download Completed!')

def get_transfer_test_video(output_dir='./data'):
    os.makedirs(output_dir, exist_ok=True)
    url = 'https://huggingface.co/datasets/Amar-S/LLNL_DSC_2025/resolve/main/Robotics_Lab_Data/transfer_test_video/VialsWithColor_Every10th_Subset.mp4'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(f'{output_dir}/VialsWithColor_Every10th_Subset.mp4', 'wb') as f:
            f.write(response.content)
        print('Transfer Test MP4 Download Completed!')
    else:
        print("Failed to download MP4 File:", response.status_code)

def get_additional_test_videos(output_dir='./data'):
    os.makedirs(output_dir, exist_ok=True)
    base_url = 'https://huggingface.co/datasets/Amar-S/LLNL_DSC_2025/resolve/main/Robotics_Lab_Data/extra_videos/'
    filenames = [
        'CV_TrialRun_1_VialsNoColor_v1.mp4',
        'CV_TrialRun_2_VialsWithColor_A2B_v1.mp4',
        'CV_TrialRun_2_VialsWithColor_B2A_v1.mp4',
        'CV_TrialRun_2_VialsWithColor_v1.mp4'
    ]

    for fname in filenames:
        url = base_url + fname
        print(f'Downloading {fname}...')
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'{output_dir}/{fname}', 'wb') as f:
                f.write(response.content)
            print(f'{fname} downloaded.')
        else:
            print(f'Failed to download {fname}, status code: {response.status_code}')

    print('Additional Test Video Files Download Completed!')

def prepare_transfer_test_dataset(
    data_dir='./data',
    clip_len=8,
    resize_shape=(256, 256),
    transform=None,
):

    print("Downloading transfer test video and frames...")
    get_transfer_test_video(output_dir=data_dir)
    get_transfer_test_frames(output_dir=data_dir)
    get_transfer_test_masks(output_dir=data_dir)

    print("Preprocessing frames...")
    preprocess_transfer_test_frames(
        input_dir=f'{data_dir}/VialsWithColor_Every10th_Subset.tar.gz',
        output_dir=f'{data_dir}/frames',
        resizeFrames=True,
        output_size=resize_shape
    )

    print("Preprocessing masks...")
    preprocess_transfer_test_mask_tensors(
        input_dir=data_dir,
        output_dir=f'{data_dir}/masks/VialsWithColor_Every10th_Subset',
        resizeMasks=True,
        size=resize_shape
    )

    frames_dir = f'{data_dir}/frames/VialsWithColor_Every10th_Subset/resized'
    masks_dir = f'{data_dir}/masks/VialsWithColor_Every10th_Subset'

    print("Creating dataset...")
    dataset = FrameMaskVideoDataset(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        clip_len=clip_len,
        transform=transform or ToTensor()
    )

    print('Transfer Test Dataset completed!')
    return dataset