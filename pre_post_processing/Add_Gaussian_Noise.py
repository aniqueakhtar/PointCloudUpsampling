import open3d as o3d
import numpy as np
import os, glob

path = './8x/'
downsampled_path = path+'0/ply/'

p = {0.50: '0_50',
     1.00: '1_00',
     1.50: '1_50',
     2.00: '2_00',
     2.50: '2_50',
     3.00: '3_00'}

files = sorted(glob.glob(downsampled_path+'**.ply'))

for percentage in p.keys():
    folder = path + p[percentage]
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder+'/ply/'):
        os.makedirs(folder+'/ply/')
    if not os.path.exists(folder+'/xyz/'):
        os.makedirs(folder+'/xyz/')
    
    for file in files:
        pcd = o3d.io.read_point_cloud(file)
        xyz = np.asarray(pcd.points)
        
        std = np.std(xyz, axis=0)
        noise = np.random.normal(0, std, size=xyz.shape) * percentage/100
        xyz_noisy = xyz+noise
        
        # save noisy_xyz
        file_name = os.path.basename(file).rsplit('.',1)[0]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(xyz_noisy)
        pcd_path = os.path.join(folder, 'ply', file_name+'.ply')
        o3d.io.write_point_cloud(pcd_path, pcd2)
        
        xyz_path = os.path.join(folder, 'xyz', file_name+'.xyz')
        np.savetxt(xyz_path, xyz_noisy)
        
        