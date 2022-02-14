import open3d as o3d
import os, glob, argparse
import numpy as np
import torch
import MinkowskiEngine as ME
from model.Network import MyNet


def parse_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # For 8x upsampling
  parser.add_argument("--pretrained", default='./ckpts/8x_0x_ks5/iter64000.pth', help='Path to pretrained model')
  # For 4x upsampling
  # parser.add_argument("--pretrained", default='./ckpts/4x_0x_ks5/iter64000.pth', help='Path to pretrained model')
  parser.add_argument("--prefix", type=str, default='8x_0x_ks5', help="prefix of checkpoints/logger, etc.")
  parser.add_argument("--up_ratio", default=8, help='Upsample Ratio')
  
  parser.add_argument("--test_dataset", default='../Data/ShapeNet/8x/')
  parser.add_argument("--save_loc", default='results_ShapeNet/')
  parser.add_argument("--last_kernel_size", type=int, default=5, help='The final layer kernel size, coordinates get expanded by this')
  
  args = parser.parse_args()
  return args


args = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyNet(last_kernel_size=args.last_kernel_size).to(device)
ckpt = torch.load(args.pretrained)
model.load_state_dict(ckpt['model'])


save_path = os.path.join(args.save_loc, args.prefix)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
path = '../Data/Test_Gaussian/8x/'
folders = ['0', '0_50', '1_00', '1_50', '2_00', '2_50', '3_00']


for fold in folders:
    print('==='*10)
    print(fold)
    print('==='*10)
    test_files = sorted(glob.glob(os.path.join(path, fold, 'ply/**.ply')))
    l = len(test_files)
    save_fold = os.path.join(save_path, fold)
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)
    for i, pc in enumerate(test_files):
        print("!!!!!!!!==========!!!!!!!!!!  FILE NUMBER: ", i+1, ' / ', l)
        file_name = os.path.basename(pc)
        print(file_name)
        pcd = o3d.io.read_point_cloud(pc)
        coords = np.asarray(pcd.points)
            
        out_list = []
        p = ME.utils.batched_coordinates([coords])
        part_T = np.random.randint(0, coords.max(), size=(coords.shape[0]*args.up_ratio, 3))
        p2 = ME.utils.batched_coordinates([part_T])
        f = torch.from_numpy(np.vstack(np.expand_dims(np.ones(p.shape[0]), 1))).float()
        x = ME.SparseTensor(feats=f, coords=p).to(device)
        
        with torch.no_grad():
            out, _, _, _ = model(x, coords_T=p2, device=device, prune=True)
            
        
        rec_pc = out.C[:,1:]
        print("Number of points in input point cloud : ", coords.shape[0])
        print("Number of points in output rec point cloud : ", rec_pc.numpy().shape[0])
        rec_pcd = o3d.geometry.PointCloud()
        rec_pcd.points = o3d.utility.Vector3dVector(rec_pc)
        recfile = os.path.join(save_fold, file_name)
        o3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

