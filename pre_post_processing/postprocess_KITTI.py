import open3d as o3d
import glob
import os
import numpy as np
# import pptk

files = sorted(glob.glob('./4x_0x_ks5/**.ply'))

for f in files:
    name = os.path.basename(f)
    ## value of k in the preprocess is saved at the end of the file name.
    num = os.path.basename(f).rsplit('.',1)[0].split('_')[-1]
    
    pcd = o3d.io.read_point_cloud(f)
    xyz = np.asarray(pcd.points)
    xyz = xyz/int(num)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud('./post_processed/'+name, pcd)