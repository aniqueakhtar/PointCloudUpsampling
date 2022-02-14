import numpy as np
import glob
import os
import open3d as o3d

orig_dir = './orig/'
dest_dir = './ply/'

files = sorted(glob.glob(orig_dir + '**.bin'))
# Voxelization / Quantization rate
## Should be optimized for best results
k = 100

for f in files:
    bin_pcd = np.fromfile(f, dtype=np.float32)
    # Reshape and drop reflection values
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    file_name = os.path.basename(f).rsplit('.',1)[0]
    
    xyz = np.unique(np.rint(points*k), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    dest = os.path.join(dest_dir, file_name+'_'+str(k)+'.ply')
    o3d.io.write_point_cloud(dest, pcd)
