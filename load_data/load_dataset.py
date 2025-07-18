#!/usr/bin/env python3

# From the provided material
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import os
import torch
from PIL import Image
import numpy as np
import random
import pandas as pd

# Define some check variables
N_SCENES_DOWNLOADED_TRAIN = 170
N_SCENES_DOWNLOADED_TEST = 39
N_TOT_FRAMES = 24
N_MAX_TRIES = 10 # for filtering - will try 10 times to find sample that is on camera

# Current prod dataloader
# This will correctly apply the modal_mask filtering to get a single object
class MOVi_Dataset(Dataset):
    def __init__(self, 
                 root,
                 split = 'train' or 'test', 
                 n_frames = 8,
                 n_samples = 1000,
                 #box_format = 'xywh'
                 ):
        """
        Initialize the MOVi dataset loader.
        This dataloader picks a random scene (video sample), a random object, and a random camera view.
        It will then load all frames from the chosen video and camera view, for its chosen object.
        
        Args:
            root: The root folder that holds the unzipped sample folders
            split: Which root subfolder to draw from (train or test)
            n_frames: How many consecutive frames to load
            n_samples: How many samples to load. This is equal to the number of objects you want to load.
            
        The dataset returned will be in the form of a dictionary, containing keys:
        'frames': RGB content, tensor with 3 channels, depth = n_frames (consecutive frames)
        'depths': modal depths, tensor, binary single channel
        'modal_masks': modal masks (occluded), binary single channel 
        'amodal_masks': amodal masks, binary single channel
        'amodal_content': RGB content, tensor with 3 channels
        'metadata': A metadata dictionary, offering info on scene, camera, object ID, as well as the number of objects in the scene.
            
        """
        print('Dataset init on', split)

        self.split = split
        self.top_dir = f'{root}/{split}/'
        print('Init data top dir:', self.top_dir)

        #self.box_format = box_format

        # Get directories in data_dir/train-test
        self.scenes = [entry for entry in os.listdir(self.top_dir) if os.path.isdir(os.path.join(self.top_dir, entry))]
        
        assert n_frames <= N_TOT_FRAMES
        self.n_frames = n_frames
        self.n_samples = n_samples

    def __len__(self):
        # In theory this could be like n_scenes*n_objects
        # To get total number of (cam-invariant) objects
        return self.n_samples

    def load_cam_frames(self, scene, 
                    cam_idx,
                    start, stop, 
                    modality):
        """
        One load-frames loads camera-level stuff (rgb, depth)
        The other one loads object-level stuff (scene/cam/obj_i/amodal_mask or content)
        """
        # Load frame range
        imgs = []
        suffix = '.png'

        totensor = ToTensor()

        for i in range(start, stop):
            # loads train/scene_id/cam_id/frames_or_depth_or_modal/frame_id.png
            if modality == 'modal_masks':
                load_file = f'{self.top_dir}/{scene}/{cam_idx}/segmentation_{str(i).zfill(5)}{suffix}'
                tens = totensor(Image.open(load_file))
            
            if modality == 'rgba_full':
                load_file = f'{self.top_dir}/{scene}/{cam_idx}/rgba_{str(i).zfill(5)}{suffix}'
                tens = totensor(Image.open(load_file).convert('RGB')) # RGB, 3 chans

            if modality == 'depth_full':
                load_file = f'{self.top_dir}/{scene}/{cam_idx}/depth_{str(i).zfill(5)}.tiff'
                tens = totensor(Image.open(load_file).convert('RGB')) # RGB, 3 chans
                
            tens = totensor(Image.open(load_file))
            imgs.append(tens)

        tensor = torch.stack(imgs, dim = 1)

        return tensor
    
    def load_obj_frames(self, scene, 
                    cam_idx,
                    object_idx,
                    start, stop, 
                    modality):
        """
        This loaded loads object-level stuff
        """
        # Load frame range
        imgs = []
        # amodal_segs, content, rgba_full, modal_masks, depth_full
        suffix = '.png'

        totensor = ToTensor()

        for i in range(start, stop):
            if modality == 'amodal_segs':
                load_file = f'{self.top_dir}/{scene}/{cam_idx}/{object_idx}/segmentation_{str(i).zfill(5)}{suffix}'
                tens = totensor(Image.open(load_file))

            if modality == 'content':
                load_file = f'{self.top_dir}/{scene}/{cam_idx}/{object_idx}/rgba_{str(i).zfill(5)}{suffix}'
                tens = totensor(Image.open(load_file).convert('RGB'))
                
            if modality == 'depth_full':
                load_file = f'{self.top_dir}/{scene}/{cam_idx}/{object_idx}/rgba_{str(i).zfill(5)}{suffix}'
                tens = totensor(Image.open(load_file).convert('RGB'))
            imgs.append(tens)

        tensor = torch.stack(imgs, dim = 1)
        return tensor


    def __getitem__(self, idx):
        # Select a random sample
        random_scene = np.random.choice(self.scenes)

        # Get the list of objects in that sample
        all_object_ids = self.all_objects(self.top_dir + random_scene + '/camera_0000/' )
        
        # Pick a random object 
        target_object_id = np.random.choice(all_object_ids)

        """
        Loading from multiple cameras in parallel:
        """

        # Make these random
        start = random.randint(0, 24-self.n_frames) # pick random integer between 0 and 24-n_frames 
        stop = start+self.n_frames # end at 

        i = random.randint(0, 5) # pick a random camera
        cam_id = f'camera_{str(i).zfill(4)}'
        frames, depths, modal_masks, amodal_segs, amodal_content = self.load_camera(random_scene, cam_id = cam_id, 
                                                                                    obj_id = target_object_id, start = start, stop = stop)
        
        # Inflate modal masks to 255
        # No need - already done in load camera!!
        # modal_masks = modal_masks*255
        # modal_masks = modal_masks.to(torch.uint8)
        obj_id_int = int(str(target_object_id).split(sep="_")[-1]) # get integer obj ID
        modal_masks = (modal_masks == obj_id_int).int() # filter into a binary modal mask for the object
        # this is one object, all frames
        # Add tracking info to the single sample
        sample = {
            'frames': frames,
            'depths': depths,
            'modal_masks': modal_masks,
            'amodal_masks': amodal_segs,
            'amodal_content': amodal_content,
            'metadata': {'scene': str(random_scene),
                         'cam_id': cam_id,
                         'obj_id': str(target_object_id),
                         'n_tot_objects_in_scene': len(all_object_ids)}
        }
        return sample

    
    def load_camera(self, scene_id, cam_id, obj_id, start, stop):

        # Load the target objects 
        modal_segs = self.load_cam_frames(scene_id, 
                                            cam_id,
                                            start, stop,
                                            'modal_masks')
        
        # Inflate modal_segs from float to integers
        # Should give one integer in range(0, Nobj-1)
        modal_segs = modal_segs*255
        modal_segs = modal_segs.int()

        # Load frames corresponding to inputs
        frames = self.load_cam_frames(scene_id, 
                                      cam_id, 
                                      start, 
                                      stop, 
                                      'rgba_full')[:-1] #drop the A 

        # Load depth (though we will have to replace with Depth-Anything-V2 estimates)
        depths = self.load_cam_frames(scene_id, cam_id, start, stop, 'depth_full')

        amodal_segs = self.load_obj_frames(scene_id, cam_id, obj_id, start, stop, 'amodal_segs')
        amodal_content = self.load_obj_frames(scene_id, cam_id, obj_id, start, stop, 'content')
        
        return frames, depths, modal_segs, amodal_segs, amodal_content
    
    def all_objects(self, pth):
        """
        Given a path, get the objects at that path using regex
        """
        #print('looking for all objects at', pth)
        
        # Find all matches
        matches = []
        for fname in sorted(os.listdir(pth)):
            if 'obj_' in fname:
                matches.append(fname)

        #print(matches)
        return matches # list of ['obj_0001', 'obj_0009',...]
    

# This is the modified function from Amar-S
# Filtering element - rejects images if the sample is completely off screen
class MOVi_Dataset_Filtered(MOVi_Dataset):
    def __init__(self, 
                 root,
                 split = 'train' or 'test', 
                 n_frames = 8,
                 n_samples = 1000,
                 foreground_ratio_thresh = 0.1
                 #box_format = 'xywh'
                 ):
        """
        Initialize the MOVi dataset loader, with Filtering.
        Inherits functions from the basic MOVi_Dataset.
        Filtering ensures the modal_mask is not all zero, i.e., the object is not fully occluded.
        This dataloader picks a random scene (video sample), a random object, and a random camera view.
        It will then load all frames from the chosen video and camera view, for its chosen object.
        
        Args:
            root: The root folder that holds the unzipped sample folders
            split: Which root subfolder to draw from (train or test)
            n_frames: How many consecutive frames to load
            n_samples: How many samples to load. This is equal to the number of objects you want to load.
            foreground_ratio_thresh: (float) minimum no of pixels in the modal mask that have to be non-zero
            
        The dataset returned will be in the form of a dictionary, containing keys:
        'frames': RGB content, tensor with 3 channels, depth = n_frames (consecutive frames)
        'depths': modal depths, tensor, binary single channel
        'modal_masks': modal masks (occluded), binary single channel 
        'amodal_masks': amodal masks, binary single channel
        'amodal_content': RGB content, tensor with 3 channels
        'metadata': A metadata dictionary, offering info on scene, camera, object ID, as well as the number of objects in the scene.
            
        """
        print('Dataset init on', split)

        self.split = split
        self.top_dir = f'{root}/{split}/'
        print('Init data top dir:', self.top_dir)

        #self.box_format = box_format

        # Get directories in data_dir/train-test
        self.scenes = [entry for entry in os.listdir(self.top_dir) if os.path.isdir(os.path.join(self.top_dir, entry))]
        
        assert n_frames <= N_TOT_FRAMES
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.foreground_ratio_thresh = foreground_ratio_thresh

    def __len__(self):
        # In theory this could be like n_scenes*n_objects
        # To get total number of (cam-invariant) objects
        return self.n_samples


    def __getitem__(self, idx):
        """Selects item only if contains something"""
        found = False
        max_tries = N_MAX_TRIES
        tries = 0
        while not found and tries < max_tries:
            # Select a random sample
            random_scene = np.random.choice(self.scenes)
            all_object_ids = self.all_objects(self.top_dir + random_scene + '/camera_0000/')
            target_object_id = np.random.choice(all_object_ids)

            start = random.randint(0, 24-self.n_frames)
            stop = start + self.n_frames
            i = random.randint(0, 5)
            cam_id = f'camera_{str(i).zfill(4)}'
            frames, depths, modal_masks, amodal_segs, amodal_content = self.load_camera(random_scene, cam_id=cam_id, 
                                                                                        obj_id=target_object_id, 
                                                                                        start=start, stop=stop)
            # Inflate modal masks to 255
            # No need - already done in load camera!!
            # modal_masks = modal_masks*255
            # modal_masks = modal_masks.to(torch.uint8)
            # Select the integer obj id
            obj_id_int = int(str(target_object_id).split(sep="_")[-1]) # get integer obj ID
            modal_masks = (modal_masks == obj_id_int).int() # filter into a binary modal mask for the object

            # Check if modal_mask contains any nonzero pixels
            # We reject the sample if the object is fully occluded (modal mask all zero)
            foreground_ratio = modal_masks.sum() / modal_masks.numel()
            # if foreground_ratio > self.foreground_ratio_thresh:
            if modal_masks.sum() > 0:
                found = True
                sample = {
                    'frames': frames,
                    'depths': depths,
                    'modal_masks': modal_masks,
                    'amodal_masks': amodal_segs,
                    'amodal_content': amodal_content,
                    'metadata': {'scene': str(random_scene),
                                'cam_id': cam_id,
                                'obj_id': str(target_object_id),
                                'n_tot_objects_in_scene': len(all_object_ids),
                                'foreground_ratio': foreground_ratio}
                }
                return sample
            else:
                tries += 1

        # If no valid sample found after max_tries
        raise RuntimeError("Failed to find a valid sample with nonzero modal_mask after {} tries".format(max_tries))
    

# Making minimal changes
# Inherit from the MOViDataset class
# pass the movi dataset (collection of many frames)
# and blow it up into a Dataset which has the tensor for each frame
# This is a dataloader ready for task 1.1
class MOVi_ImageDataset(MOVi_Dataset):
    """
    Loads the MOVi dataset from file, casting it in a form ready for Image Models.
    One frame per sample.
    Inherits from the MOVi_Dataset class - so requires the same args.
    Example usage:
    image_ds = MOVi_ImageDataset(root=ROOT_PATH, split = 'test', n_frames = 8, n_samples=30)
    """
    def __init__(self, n_cameras=6,nframe_sample=24,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_indices = []
        # Build a list of (scene, camera, object, frame_idx) for all frames
        for scene in self.scenes:
            for cam_id in [f'camera_{str(i).zfill(4)}' for i in range(n_cameras)]:  # assuming 6 cameras
                obj_path = os.path.join(self.top_dir, scene, cam_id)
                if not os.path.exists(obj_path):
                    continue
                object_ids = self.all_objects(obj_path)
                for obj_id in object_ids:
                    # Assume all videos have 24 frames (0..24)
                    for frame_idx in range(nframe_sample):  # or use dynamic length if needed
                        self.frame_indices.append((scene, cam_id, obj_id, frame_idx))
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        scene, cam_id, obj_id, frame_idx = self.frame_indices[idx]
        # Load a single frame for each modality
        frame = self.load_cam_frames(scene, cam_id, frame_idx, frame_idx+1, 'rgba_full').squeeze(1)[:-1] # ensure we get RGB 
        depth = self.load_cam_frames(scene, cam_id, frame_idx, frame_idx+1, 'depth_full').squeeze(1)
        modal_mask = self.load_cam_frames(scene, cam_id, frame_idx, frame_idx+1, 'modal_masks').squeeze(1)
        amodal_mask = self.load_obj_frames(scene, cam_id, obj_id, frame_idx, frame_idx+1, 'amodal_segs').squeeze(1)
        amodal_content = self.load_obj_frames(scene, cam_id, obj_id, frame_idx, frame_idx+1, 'content').squeeze(1)
        
        modal_mask = (modal_mask * 255).to(torch.uint8)
        
        sample = {
            'frame': frame,
            'depth': depth,
            'modal_mask': modal_mask,
            'amodal_mask': amodal_mask,
            'amodal_content': amodal_content,
            'scene': scene,
            'cam_id': cam_id,
            'obj_id': obj_id,
            'frame_idx': frame_idx
        }
        return sample
    

# Existing dataset class inherited
# This extension
class MOVi_Dataset_SmartSelect(MOVi_Dataset):
    def __init__(self, 
                 root,
                 split = 'train' or 'test', 
                 n_frames = 8,
                 n_samples = 100,
                 min_modal_pixels_visible = 10,
                 min_occ_rate = 0.5,
                 max_occ_rate = 0.9,
                 #box_format = 'xywh'
                 ):
        """
        Initialize the MOVi dataset loader, with Dataset Smart selection.
        Inherits functions from the basic MOVi_Dataset.
        Smart Selection works by ensuring certain conditions are met: modal_mask is not all zero, i.e., the object is not fully occluded,
        the occlusion rate is high enough to learn something meaningful about completion etc. This uses dataset metadata.
        This dataloader picks a random scene (video sample), a random object, and a random camera view.
        It will then load all frames from the chosen video and camera view, for its chosen object.
        
        Args:
            root: The root folder that holds the unzipped sample folders
            split: Which root subfolder to draw from (train or test)
            n_frames: How many consecutive frames to load. Defaults to 8.
            n_samples: How many samples to load. This is equal to the number of objects you want to load. Defaults to 100.
            min_modal_pixels_visible (int): How many object pixels have to be visible in the modal image to be accepted. In the case of video,
                this criterion is applied on the average number of modal pixels visible. Defaults to 10.
            min_occ_rate (float): Minimum object occlusion rate. This the ratio of modal pixels/number of amodal pixels. 0 to 1,
                defaults to 0.5. For videos, this is the average across N_FRAMES.
            max_occ_rate (float): Maximum object occlusion rate to accept. 0 to 1, defaults to 0.9. Must be bigger than min_occ_rate.
            
        The dataset returned will be in the form of a dictionary, containing keys:
        'frames': RGB content, tensor with 3 channels, depth = n_frames (consecutive frames)
        'depths': modal depths, tensor, binary single channel
        'modal_masks': modal masks (occluded), binary single channel 
        'amodal_masks': amodal masks, binary single channel
        'amodal_content': RGB content, tensor with 3 channels
        'metadata': A metadata dictionary, offering info on scene, camera, object ID, as well as the number of objects in the scene.
            
        """
        print('Dataset init on', split)

        self.split = split
        self.top_dir = f'{root}/{split}/'
        print('Init data top dir:', self.top_dir)

        #self.box_format = box_format

        # Get directories in data_dir/train-test
        self.scenes = [entry for entry in os.listdir(self.top_dir) if os.path.isdir(os.path.join(self.top_dir, entry))]
        
        assert n_frames <= N_TOT_FRAMES
        self.n_frames = n_frames
        self.n_samples = n_samples

        assert min_occ_rate < max_occ_rate
        
        # Perform dataset selection
        self.image = False
        if n_frames == 1:
            # Load image metadata
            df = pd.read_csv(f"../dataset_metadata/{split}_per_frame_metadata.csv", index_col=0)
            # Apply criteria
            df_select = df[(df['scene_id'].isin(self.scenes)) &
                        (df['occ_rate'] >= min_occ_rate) &
                        (df['occ_rate'] <= max_occ_rate) &
                        (df['modal_n_pixels_foreground'] >= min_modal_pixels_visible)].copy().reset_index(drop=True)
            self.image = True
        else:
            # loading video data
            self.image = False
            df = pd.read_csv(f"../dataset_metadata/{split}_video_metadata.csv", index_col=0)
            # Apply criteria
            df_select = df[(df['scene_id'].isin(self.scenes)) &
                        (df['avg_occ_rate'] >= min_occ_rate) &
                        (df['avg_occ_rate'] <= max_occ_rate) &
                        (df['avg_modal_n_pixels_foreground'] >= min_modal_pixels_visible)].copy().reset_index(drop=True)
            
        n_available = len(df_select)
        if n_available < n_samples:
            print("WARNING: Your requested n_samples is smaller than the number of samples meeting your criteria. Chance of getting duplicates.")
        self.df_select = df_select


    def __len__(self):
        # In theory this could be like n_scenes*n_objects
        # To get total number of (cam-invariant) objects
        return self.n_samples


    def __getitem__(self, idx):
        """
        Selects item only if it fulfils the criteria
        """
        df = self.df_select.copy()
        # Select a random scene - camera - object sample from the selected
        # this is now a dictionary
        random_sample = df.iloc[np.random.choice(df.index)]

        # Write out scene - camera - obj choice
        random_scene = random_sample['scene_id']
        # Random camera - all scenes should have cam 0000 to cam 0005
        cam_id = random_sample['cam_id']
        target_object_id = 'obj_{}'.format(str(random_sample['obj_id']).zfill(4))

        # Now select frames
        if self.image:
            # Select a single random frame fulfiling the criteria
            start = random_sample['frame_id']
            stop = start + 1
        else:
            # Select random frames for video
            start = random.randint(0, 24-self.n_frames)
            stop = start + self.n_frames
        
        frames, depths, modal_masks, amodal_segs, amodal_content = self.load_camera(random_scene, cam_id=cam_id, 
                                                                                    obj_id=target_object_id, 
                                                                                    start=start, stop=stop)
        # Inflate modal masks to 255
        # No need - already done in load camera!!
        # modal_masks = modal_masks*255
        # modal_masks = modal_masks.to(torch.uint8)
        # Select the integer obj id
        obj_id_int = int(str(target_object_id).split(sep="_")[-1]) # get integer obj ID
        modal_masks = (modal_masks == obj_id_int).int() # filter into a binary modal mask for the object

        # Check if modal_mask contains any nonzero pixels
        # We reject the sample if the object is fully occluded (modal mask all zero)
        sample = {
            'frames': frames,
            'depths': depths,
            'modal_masks': modal_masks,
            'amodal_masks': amodal_segs,
            'amodal_content': amodal_content,
            'metadata': random_sample.to_dict()
        }
        return sample
    
    