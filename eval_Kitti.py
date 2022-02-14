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
  # parser.add_argument("--pretrained", default='./ckpts/8x_0x_ks5/iter64000.pth', help='Path to pretrained model')
  # For 4x upsampling
  parser.add_argument("--pretrained", default='./ckpts/4x_0x_ks5/iter64000.pth', help='Path to pretrained model')
  parser.add_argument("--prefix", type=str, default='4x_0x_ks5', help="prefix of checkpoints/logger, etc.")
  parser.add_argument("--up_ratio", default=4, help='Upsample Ratio')
  
  parser.add_argument("--test_dataset", default='../Data/Kitti/')
  parser.add_argument("--save_loc", default='results_Kitti/')
  parser.add_argument("--last_kernel_size", type=int, default=5, help='The final layer kernel size, coordinates get expanded by this')
  
  args = parser.parse_args()
  return args


def kdtree_partition(pc, max_num):
    parts = []
    
    class KD_node:  
        def __init__(self, point=None, LL = None, RR = None):  
            self.point = point  
            self.left = LL  
            self.right = RR
            
    def createKDTree(root, data):
        if len(data) <= max_num:
            parts.append(data)
            return
        
        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]
        
        point = data_sorted[int(len(data)/2)]  
        
        root = KD_node(point)  
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))])  
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):]) 
        
        return root
    
    init_root = KD_node(None)
    _ = createKDTree(init_root, pc)  
    return parts

args = parse_args()
test_files = glob.glob(args.test_dataset+'*.ply')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyNet(last_kernel_size=args.last_kernel_size).to(device)
ckpt = torch.load(args.pretrained)
model.load_state_dict(ckpt['model'])

l = len(test_files)

save_path = os.path.join(args.save_loc, args.prefix)
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i, pc in enumerate(test_files):
    print("!!!!!!!!==========!!!!!!!!!!  FILE NUMBER: ", i+1, ' / ', l)
    file_name = os.path.basename(pc)
    pcd = o3d.io.read_point_cloud(pc)
    coords = np.asarray(pcd.points)
    
    parts = kdtree_partition(coords, 70000)
    
    out_list = []
    for j,part in enumerate(parts):
        p = ME.utils.batched_coordinates([part])
        part_T = np.random.randint(0, part.max(), size=(part.shape[0]*args.up_ratio, 3))
        p2 = ME.utils.batched_coordinates([part_T])
        f = torch.from_numpy(np.vstack(np.expand_dims(np.ones(p.shape[0]), 1))).float()
        x = ME.SparseTensor(feats=f, coords=p).to(device)
        
        with torch.no_grad():
            out, _, _, _ = model(x, coords_T=p2, device=device, prune=True)
            
        out_list.append(out.C[:,1:])
      
    rec_pc = torch.cat(out_list, 0)
    print("Number of points in input point cloud : ", coords.shape[0])
    print("Number of points in output rec point cloud : ", rec_pc.numpy().shape[0])
    rec_pcd = o3d.geometry.PointCloud()
    rec_pcd.points = o3d.utility.Vector3dVector(rec_pc)
    recfile = os.path.join(save_path, file_name)
    o3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)
