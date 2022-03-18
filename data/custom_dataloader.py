import os
import numpy as np
import cv2
from PIL import Image
import random
import torch
from torchvision import transforms
import torch.utils.data as data

class CustomImgDataset(data.Dataset):
  def __init__(self, H, W, base_path='/vision/u/ajarno/cs231a/custom_data/'):
    self.H = H
    self.W = W
    self.data_path = os.path.join(base_path, 'images')   # Location at which frames are stored
    self.down_path = os.path.join(base_path, 'downsampled_images')
    
    if not os.path.exists(self.down_path):
        os.mkdir(self.down_path)
    if not os.path.exists(self.data_path):
        os.mkdir(self.data_path)
    if not os.path.exists(os.path.join(base_path, "depth")):
        os.mkdir(os.path.join(base_path, "depth"))
    if not os.path.exists(os.path.join(base_path, "VNL_depth")):
        os.mkdir(os.path.join(base_path, "VNL_depth"))
    if not os.path.exists(os.path.join(base_path, "refined_depth")):
        os.mkdir(os.path.join(base_path, "refined_depth"))
    if not os.path.exists(os.path.join(base_path, "json_file")):
        os.mkdir(os.path.join(base_path, "json_file"))  
    self.data = self.collect_filenames()

  def preprocessing_transforms(self):
    return transforms.Compose([
        ToTensor(mode="test")
    ])


  def collect_filenames(self):
    print("CustomImgDataset Loading...")
    ret = []
    for d in sorted(os.listdir(self.data_path)):
        full_path = os.path.join(self.data_path, d)
        if os.path.isfile(full_path):
            ret += [full_path]
    print("CustomImgDataset loaded! (", len(ret), " images loaded )")
    return ret

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    fn = self.data[idx]
    alt = fn.replace("images", 'downsampled_images')
    if os.path.exists(alt):
        image = Image.open(alt).resize((self.W, self.H), Image.NEAREST)
    else:
        image = Image.open(fn).resize((self.W, self.H), Image.NEAREST)
    image.save(self.down_path + "/" + fn.split("/")[-1])
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = image[None,:]
    
    out = self.preprocessing_transforms()({'image': image, 
            'focal': torch.tensor([519]),
            'image_path': [fn],
            'gt_depth_path': [""],
            'rawD': [""],
            'mirror_instance_mask_path': [""],
            })
    out['image'] = out['image'][None,:]
    return out
