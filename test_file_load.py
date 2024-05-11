import os
import numpy as np

dir = "output/mesh/np_savez.npz"
data = np.load(dir)

if not os.path.exists(dir): # ディレクトリが存在するか確認
    print("not exist")
else:
    meshes = data['arr_0'] 
    faces = data['arr_1']
    camera_pose = data['arr_2']
    camera_center = data['arr_3']
    fx = data['arr_4']
    fy = data['arr_5']
    cx = data['arr_6']
    cy = data['arr_7']
    zfar = data['arr_8']


print("done to load")