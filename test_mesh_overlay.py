import os
import numpy as np

import pyrender

import cv2
import torch

# cuda0 = torch.device('cuda:0')
image0 = cv2.imread("/home/initial/workspace/homer_ws/src/homer/data/camera_head.png")
# image = torch.tensor(image, device=cuda0)
# image = image * torch.tensor([0.229, 0.224, 0.225], device=image.device).reshape(3,1,1)
# image = image + torch.tensor([0.485, 0.456, 0.406], device=image.device).reshape(3,1,1)
# image = image.permute(1, 2, 0).cpu().numpy()
image = cv2.imread("output/mesh/image.png")
# image = torch.tensor(image, device=cuda0)
dir = "output/mesh/np_savez.npz"
data = np.load(dir)

if not os.path.exists(dir): # ディレクトリが存在するか確認
    print("not exist")
else:
    mesh = data['arr_0'] 
    face = data['arr_1']
    camera_pose = data['arr_2']
    camera_center = data['arr_3']
    fx = data['arr_4']
    fy = data['arr_5']
    cx = data['arr_6']
    cy = data['arr_7']
    zfar = data['arr_8']
    bbox = data['arr_9']
    keyp = data['arr_10']


###
bbox = bbox[0]
pt1 = (int(bbox[0]), int(bbox[1]))
pt2 = (int(bbox[2]), int(bbox[3]))
line_type = cv2.LINE_AA
line_color = (255, 0, 255) # マゼンタ
thickness = 1
cv2.rectangle(image0, pt1, pt2, line_color, thickness, line_type)
cv2.imshow("image0", image0)
cv2.waitKey(1)
center = (bbox[2:4] + bbox[0:2]) / 2.0
print(center)
###

print("done to load")
from typing import List, Optional

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

###
mesh_base_color=(1.0, 1.0, 0.9)
material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

import trimesh

mesh = trimesh.Trimesh(mesh, face)
# rot = trimesh.transformations.rotation_matrix(
#     np.radians(180), [1, 0, 0])
# mesh.apply_transform(rot)
mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
###
scene_bg_color=(0,0,0)
light_nodes = create_raymond_lights()
scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
scene.add(mesh, 'mesh')

# camera_pose = np.eye(4)
# camera_pose[:3, 3] = camera_translation
# camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy,
                                   cx=cx, cy=cy, zfar=zfar)
scene.add(camera, pose=camera_pose)

import pyrender
renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
                                              
for node in light_nodes:
    scene.add_node(node)

color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
color = color.astype(np.float32) / 255.0
renderer.delete()

# if return_rgba:
#     return color

valid_mask = (color[:, :, -1])[:, :, np.newaxis]
side_view=False
# if not side_view:
output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
# else:
# output_img = color[:, :, :3]

output_img = output_img.astype(np.float32)
cv2.imshow("", output_img)
cv2.waitKey(0)