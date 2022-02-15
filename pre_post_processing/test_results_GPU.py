import glob
import open3d as o3d
import numpy as np
import os
from utils.pc_error_wrapper import pc_error
from pytorch3d.loss import chamfer_distance
import torch
import pandas as pd

orig_pc = '/home/anique/Upsampling_3/Data/8i_test/orig/'
input_pc = '../Data/8i_test/8x/'
output_pc = 'results/8x_0x_ks5/'
csv_name = 'test_8x_ks5.csv'

files = sorted(glob.glob(input_pc+'*.ply'))
reso = 1024

device = torch.device('cuda')
for i, m in enumerate(files):
    print("!!!!!!!!==========!!!!!!!!!!  FILE NUMBER: ", i)
    results = {}
    file_name = os.path.basename(m)
    pcd = o3d.io.read_point_cloud(m)
    xyz_in = np.asarray(pcd.points)
    infile = m
    
    pcd = o3d.io.read_point_cloud(output_pc + file_name)
    xyz_out = np.asarray(pcd.points)
    outfile = output_pc + file_name
    
    GTfile = orig_pc + file_name
    pcd = o3d.io.read_point_cloud(GTfile)
    xyz_GT = np.asarray(pcd.points)
    
    
    print("Number of points in input point cloud : ", xyz_in.shape[0])
    print("Number of points in output rec point cloud : ", xyz_out.shape[0])
    print("Number of points in GT point cloud : ", xyz_GT.shape[0])
    
    
    print('==========  Measuring PSNR')
    
    pc_error_metrics_in = pc_error(infile1=GTfile, infile2=infile, res=reso)
    print(pc_error_metrics_in["mseF,PSNR (p2point)"][0])
    
    pc_error_metrics_out = pc_error(infile1=GTfile, infile2=outfile, res=reso)
    print(pc_error_metrics_out["mseF,PSNR (p2point)"][0])
    
    print('==========  Measuring CD')
    
    pc_out = torch.tensor(xyz_out).unsqueeze(0).float().to(device)
    pc_in = torch.tensor(xyz_in).unsqueeze(0).float().to(device)
    pc_GT = torch.tensor(xyz_GT).unsqueeze(0).float().to(device)
    
    loss_chamfer_out, _ = chamfer_distance(pc_GT, pc_out)
    print(loss_chamfer_out)

    loss_chamfer_in, _ = chamfer_distance(pc_GT, pc_in)
    print(loss_chamfer_in)
    
    
    results["MSE_F_in"] = pc_error_metrics_in["mseF      (p2point)"][0]
    results["MSE_F_out"] = pc_error_metrics_out["mseF      (p2point)"][0]
    results["MSE_F, PSNR_in"] = pc_error_metrics_in["mseF,PSNR (p2point)"][0]
    results["MSE_F, PSNR_out"] = pc_error_metrics_out["mseF,PSNR (p2point)"][0]
    results["Hausdorff_F_in"] = pc_error_metrics_in["h.        (p2point)"][0]
    results["Hausdorff_F_out"] = pc_error_metrics_out["h.        (p2point)"][0]
    results["Hausdorff_F, PSNR_in"] = pc_error_metrics_in["h.,PSNR   (p2point)"][0]
    results["Hausdorff_F, PSNR_out"] = pc_error_metrics_out["h.,PSNR   (p2point)"][0]
    results["Chamfer_in"] = loss_chamfer_in.cpu().numpy()
    results["Chamfer_out"] = loss_chamfer_out.cpu().numpy()
    
    results = pd.DataFrame([results])
    
    if i == 0:
        all_results = results.copy(deep=True)
    else:
        all_results = all_results.append(results, ignore_index=True)
    
os.system('rm *.ply')
all_results.to_csv(csv_name, index=False)

