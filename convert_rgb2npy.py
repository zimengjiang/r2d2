import numpy as np
import argparse
import os
from pathlib import Path
import torch.nn.functional as f

import pdb
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tvf

tensor_rgb = tvf.Compose([tvf.ToTensor()])

def normalize_(x, dim):
    x_norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    mask = x_norm == 0.0
    x_norm = x_norm + mask * 1e-16
    return x / x_norm

def check_feature_dims(root_to_current_dataset):
    all_features = os.listdir(root_to_current_dataset)
    for f in all_features:
        f_load = np.load(Path(root_to_current_dataset, f))
        print(f"\nChecking {Path(root_to_current_dataset, f)}")
        print(f"\nC x H x W: {f_load.shape}")

def convert_normalized_rgb2npy(root_to_current_dataset, image_folder, save_folder):
    all_fimgs = os.listdir(root_to_current_dataset + '/' + image_folder)
    root_to_save = root_to_current_dataset + '/' +  save_folder
    if not os.path.exists(root_to_save):
        os.makedirs(root_to_save)
    for fimg in all_fimgs:
        img = Image.open(Path(root_to_current_dataset + '/' + image_folder, fimg)).convert('RGB')
        img = torch.tensor(np.array(img),dtype=torch.float32)
        img_norm = normalize_(img,-1) # H x W x C
        img_norm_array = np.array(img_norm.reshape(img_norm.shape[0], img_norm.shape[1]*img_norm.shape[2]),dtype=np.float32) # float color texture in CUDA is 4-byte float. 
        outpath = Path(root_to_save,fimg[:-4])
        print(f"Saving features to {outpath}")
        np.save(outpath, img_norm_array)

        

def test_toy_normalize_and_reshape():
    a = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)
    a_ = 2 * torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)
    print(f"1. shape of 'a': H x W =  {a.shape[0]} x {a.shape[1]} \n",a)
    b = torch.stack((a,a_),dim=-1)
    print("2. stack 'a' and 'a_' to get 'b' \n")
    print(f"shape of 'b': H x W x C=  {b.shape[0]} x {b.shape[1]} x {b.shape[2]} \n",b)
    print("3. normalize 'b' to get 'b_norm' \n")
    b_norm = normalize_(b,-1)
    print(b_norm)
    print("4. reshpe 'b_norm' to get 'c' \n")
    c = b_norm.reshape(b_norm.shape[0],b_norm.shape[1]*b_norm.shape[2])
    print(f"shape of 'c': H x (WC) = {c.shape[0]} x {c.shape[1]}\n")
    print(c)

def test_toy_reshape():
    a = torch.tensor([[1,2,3],[4,5,6]])
    print(f"1. shape of 'a': H x W =  {a.shape[0]} x {a.shape[1]} \n",a)
    b = torch.stack((a,a),dim=-1)
    print("2. stack 'a' twice to get 'b' \n")
    print(f"shape of 'b': H x W x C=  {b.shape[0]} x {b.shape[1]} x {b.shape[2]} \n",b)
    print("3. reshpe 'b' to get 'c' \n")
    c = b.reshape(b.shape[0],b.shape[1]*b.shape[2])
    print(f"shape of 'c': H x (WC) = {c.shape[0]} x {c.shape[1]}\n")
    print(c)





if __name__ == '__main__':
    # test_toy_reshape()
    # test_toy_normalize_and_reshape()
    parser = argparse.ArgumentParser("normalize the image and save as 2D array")
    parser.add_argument("--datalist", type=str, required=True, help='path to the txt file that contains the datasets to check')
    parser.add_argument("--image_folder", type=str, required=True, help='name of the foder containing rgb images')
    parser.add_argument("--save_folder", type=str, required=True, help='name of the folder o save converted numpy arrays')
    args = parser.parse_args()
    all_datasets = np.loadtxt(args.datalist,dtype=str)
    root = '/local/home/zjiang/data/eth3d/training'
    for dataset in all_datasets:
        root_to_current_dataset = root + '/' + dataset
        print(f"\nLoading images from {root_to_current_dataset}")
        convert_normalized_rgb2npy(root_to_current_dataset,args.image_folder, args.save_folder)