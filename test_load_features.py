import numpy as np
import argparse
import os
from pathlib import Path

def check_feature_dims(root_to_current_dataset):
    all_features = os.listdir(root_to_current_dataset)
    for f in all_features:
        f_load = np.load(Path(root_to_current_dataset, f))
        print(f"\nChecking {Path(root_to_current_dataset, f)}")
        print(f"\nC x H x W: {f_load.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate the dimension of a given feauture map")
    parser.add_argument("--datalist", type=str, required=True, help='path to the txt file that contains the datasets to check')
    parser.add_argument("--feature_type", type=str, required=True, help='name of the foder containing feature maps')
    args = parser.parse_args()
    all_datasets = np.loadtxt(args.datalist,dtype=str)
    root = '/local/home/zjiang/data/eth3d/training'
    for dataset in all_datasets:
        root_to_current_dataset = root + '/' + dataset + '/' + args.feature_type
        print(f"\nLoading features from {root_to_current_dataset}")
        check_feature_dims(root_to_current_dataset)
    