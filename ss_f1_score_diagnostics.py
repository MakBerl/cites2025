import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from libs.glorys_dataset import OceanEddyDataset
from libs.unet import UNet
from libs.parse_args_stage1_diagnostics import parse_args
import sys
from tqdm import tqdm
from libs.mean_f1score import global_f1_score
from openCV_intersection import global_ss_f1_score
import pickle
from libs.ServiceDefs import EnsureDirectoryExists
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    curr_run = args.run_name
        
    base_data_dir = args.src_dir
    out_base_dir = args.dst_dir
    out_diagnostics_dir = os.path.join(out_base_dir, 'diagnostics', args.run_name)
    EnsureDirectoryExists(out_diagnostics_dir)

    index_file_path = os.path.join(out_base_dir, 'index_gl.json')
    index_data = {"years": {}}
    for year in range(1995, 1995 + 1):
        index_data["years"][str(year)] = {
            "vort": os.path.join(base_data_dir, f"{year}/vort.npy"),
            "ssh_anom": os.path.join(base_data_dir, f"{year}/ssh_anom.npy"),
            "masked_auto": os.path.join(base_data_dir, f"{year}/masked_auto.npy")
        }
    with open(index_file_path, 'w') as f:
        json.dump(index_data, f)
    print(f"Created index file for the dataset: {index_file_path}")
    
    years = [str(y) for y in range(1995, 1995 + 1)]
    dataset = OceanEddyDataset(index_file_path, years, augment=True)
    
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)
    
    model = UNet(n_channels=2, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.snapshot))
    model.eval()

    thresholds = np.linspace(0.05, 0.95, 100)
    f1_scores = {}
    #!
    ss_f1_scores={}
    filter_th=None
    #!
    batch_counter = 0

    pbar = tqdm(total=len(loader), desc="Inference")
    for batch_idx, (inputs, masks, valid_masks) in enumerate(loader):
        inputs, masks, valid_masks = inputs.to(device), masks.to(device), valid_masks.to(device)
        batch_counter += 1

        outputs = model(inputs) * valid_masks
        for threshold in thresholds:
            f1 = global_f1_score(outputs, masks * valid_masks, threshold=threshold)
            #!
            ss_f1,filter_th_arr = global_ss_f1_score(outputs, masks * valid_masks, threshold=threshold) #Среднее за батч
            print(ss_f1.shape, filter_th_arr.shape)
            ss_f1_scores[threshold] = ss_f1_scores.get(threshold, 0) + ss_f1
            #!
            f1_scores[threshold] = f1_scores.get(threshold, 0) + f1
        
        #pbar.set_postfix({'f1': float(f1_scores[thresholds[-1]] / len(loader))})
        pbar.update(1)
    pbar.close()
    #print(ss_f1_scores)
    for threshold in thresholds:
        f1_scores[threshold] /= batch_counter
        ss_f1_scores[threshold] /= batch_counter
    with open(os.path.join(out_diagnostics_dir, 'f1_scores.pkl'), 'wb') as f:
        pickle.dump(f1_scores, f)

    with open(os.path.join(out_diagnostics_dir, 'ss_f1_scores.pkl'), 'wb') as f:
        pickle.dump(ss_f1_scores, f)    
    print(f"F1 scores saved to {os.path.join(out_diagnostics_dir, 'f1_scores.pkl')}")
    print(f"F1 scores saved to {os.path.join(out_diagnostics_dir, 'ss_f1_scores.pkl')}")

    f1_scores = np.array(list(f1_scores.items()))

    orig_th_bin_arr = np.array(list(ss_f1_scores.keys()))
    ss_f1_scores_mtx = np.array(list(ss_f1_scores.values()))
    #ss_f1_scores = np.array(list(ss_f1_scores.items()))
    print('ss_f1_shape',ss_f1_scores_mtx.shape)
    plt.scatter(f1_scores[:,0], f1_scores[:,1])
    plt.xlabel('Bin threshold')
    plt.ylabel('F1 score')
    plt.savefig(os.path.join(out_diagnostics_dir, 'f1_scores.png'))
    print(f"F1 scores plot saved to {os.path.join(out_diagnostics_dir, 'f1_scores.png')}")
    plt.close()

    X,Y=np.meshgrid(filter_th_arr,orig_th_bin_arr)
    plt.pcolormesh(X,Y,ss_f1_scores_mtx)
    plt.xlabel('Bin threshold')
    plt.ylabel('Filter th')
    plt.savefig(os.path.join(out_diagnostics_dir,'ss_f1_scores.png'))

    f1_scores_max = f1_scores[f1_scores[:,1].argmax()][1]
    iou_optimal_threshold = f1_scores[f1_scores[:,1].argmax()][0]
    mean_f1_score_over_iou = np.trapz(f1_scores[:,1], f1_scores[:,0])/(f1_scores[:,0].max() - f1_scores[:,0].min())
    print(f"Max F1 score: {f1_scores_max:.4f}")
    print(f"Max F1 score threshold: {iou_optimal_threshold:.4f}")
    print(f"Mean F1 score over IoU: {mean_f1_score_over_iou:.4f}")
    
    with open(os.path.join(out_diagnostics_dir, 'summary_diagnostics.txt'), 'w') as f:
        f.write(f"Max F1 score: {f1_scores_max:.4f}\n")
        f.write(f"Max F1 score threshold: {iou_optimal_threshold:.4f}\n")
        f.write(f"Mean F1 score over IoU: {mean_f1_score_over_iou:.4f}\n")


if __name__ == "__main__":
    main()
