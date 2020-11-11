# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use


import os, pdb
from PIL import Image
import numpy as np
import torch

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
from pathlib import Path


def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1024, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores

def extract_and_save_byscale( net, img, detector, scale_f):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    s = 1.0/scale_f # current scale factor
    # down-scale the image for next iteration
    nh, nw = round(H*s), round(W*s)
    img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # if verbose: print(f"extracting at scale x{s:.02f} = {W:4d}x{H:3d}")
    # extract descriptors
    with torch.no_grad():
        res = net(imgs=[img])
        
    # get output and reliability map
    descriptors = res['descriptors'][0]
    # reliability = res['reliability'][0]
    # repeatability = res['repeatability'][0]
    return descriptors

def extract_keypoints(args, root_to_current_dataset, scale_f):
    iscuda = common.torch_set_gpu(args.gpu)
    output_folder = 'r2d2_features_' + str(scale_f)
    rgb_folder = 'rgb.txt'
    root_to_save_features = root_to_current_dataset+ '/' + output_folder
    if not os.path.exists(root_to_save_features):
        os.makedirs(root_to_save_features)
    # load the network...
    net = load_network(args.model)
    if iscuda: net = net.cuda()

    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr = args.reliability_thr, 
        rep_thr = args.repeatability_thr)
    
    imgs_under_current_folder = np.loadtxt(Path(root_to_current_dataset,rgb_folder), dtype=str)
    for img in imgs_under_current_folder:
        img_path = img[-1]
        img_name = img[-1].split('/')[-1]
        # print(f"\nExtracting features for {img_path}")
        img = Image.open(Path(root_to_current_dataset, img_path)).convert('RGB')
            
        img = norm_RGB(img)[None] 
        if iscuda: img = img.cuda()
        
        # extract keypoints/descriptors for a single image
        desc  = extract_and_save_byscale( net, img, detector, scale_f)

        desc = desc.cpu().numpy().squeeze()
        
        outpath = Path(root_to_save_features,img_name[:-4])
        
        print(f"Saving features to {outpath}")
        np.save(outpath, desc)

        #np.savez(open(outpath,'wb'), 
         #   imsize = (W,H),
          #  keypoints = xys[idxs], 
           # descriptors = desc[idxs], 
            #scores = scores[idxs])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--model", type=str, required=True, help='model path')
    
    parser.add_argument("--images", type=str, nargs='+', help='images / list')
    parser.add_argument("--tag", type=str, default='r2d2', help='output file tag')
    
    parser.add_argument("--top-k", type=int, default=5000, help='number of keypoints')

    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    
    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)

    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='use -1 for CPU')

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--scale", type=int, help='the factor to downscale the image')

    args = parser.parse_args()

    all_datasets = np.loadtxt(args.datasets,dtype=str)
    root = '/local/home/zjiang/data/eth3d/training'
    for dataset in all_datasets:
        root_to_current_dataset = root + '/' + dataset
        extract_keypoints(args, root_to_current_dataset, args.scale)

