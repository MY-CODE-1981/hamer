import os
import numpy as np
import pyrender
import cv2
import torch
import trimesh

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

##############
# from .render_openpose import render_openpose
# render_hand_keypoints(img, right_hand_keypoints, threshold=0.1, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0)
from hamer.utils.render_openpose import render_openpose
gt_keypoints_img = render_openpose(image0, keyp) / 255.
cv2.imshow("0", gt_keypoints_img)
# cv2.waitKey(0)
##############


dir_color = "output/mesh/np_savez_color.npz"
data_color = np.load(dir_color)

if not os.path.exists(dir_color): # ディレクトリが存在するか確認
    print("not exist")
else:
    verts = data_color['arr_0']
    cam_t = data_color['arr_1']
    is_right = data_color['arr_2']
    img_size = data_color['arr_3']
    n = data_color['arr_4']
    model_mano_faces = data_color['arr_5']
    device = torch.device('cuda')          # GPUデバイスを指定
    img_size = torch.from_numpy(img_size)
    img_size = img_size.to('cuda')

###
bbox = bbox[0]
pt1 = (int(bbox[0]), int(bbox[1]))
pt2 = (int(bbox[2]), int(bbox[3]))
line_type = cv2.LINE_AA
line_color = (255, 0, 255) # マゼンタ
thickness = 1
cv2.rectangle(gt_keypoints_img, pt1, pt2, line_color, thickness, line_type)
cv2.imshow("2", gt_keypoints_img)
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

all_verts = []
all_cam_t = []
all_right = []

all_verts.append(verts)
all_cam_t.append(cam_t)
all_right.append(is_right)

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

FOCAL_LENGTH = 5000
IMAGE_SIZE = 256

scaled_focal_length = FOCAL_LENGTH / IMAGE_SIZE * img_size.max()

misc_args = dict(
    mesh_base_color=LIGHT_BLUE,
    scene_bg_color=(1, 1, 1),
    focal_length=scaled_focal_length,
)

faces = model_mano_faces
# faces_new = np.array([[92, 38, 234],
#                       [234, 38, 239],
#                       [38, 122, 239],
#                       [239, 122, 279],
#                       [122, 118, 279],
#                       [279, 118, 215],
#                       [118, 117, 215],
#                       [215, 117, 214],
#                       [117, 119, 214],
#                       [214, 119, 121],
#                       [119, 120, 121],
#                       [121, 120, 78],
#                       [120, 108, 78],
#                       [78, 108, 79]])
# faces = np.concatenate([faces, faces_new], axis=0)

def vertices_to_trimesh(vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), rot_axis=[1,0,0], rot_angle=0, is_right=1):
    global faces
    # material = pyrender.MetallicRoughnessMaterial(
    #     metallicFactor=0.0,
    #     alphaMode='OPAQUE',
    #     baseColorFactor=(*mesh_base_color, 1.0))
    vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
    # if is_right:
    mesh = trimesh.Trimesh(vertices.copy() + camera_translation, faces.copy(), vertex_colors=vertex_colors)
    # else:
    #     mesh = trimesh.Trimesh(vertices.copy() + camera_translation, faces_left.copy(), vertex_colors=vertex_colors)
    # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
    
    rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
    mesh.apply_transform(rot)

    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    return mesh


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    
def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def add_point_lighting(scene, cam_node, color=np.ones(3), intensity=1.0):
    # from phalp.visualize.py_renderer import get_light_poses
    light_poses = get_light_poses(dist=0.5)
    light_poses.append(np.eye(4))
    cam_pose = scene.get_pose(cam_node)
    for i, pose in enumerate(light_poses):
        matrix = cam_pose @ pose
        # node = pyrender.Node(
        #     name=f"light-{i:02d}",
        #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
        #     matrix=matrix,
        # )
        node = pyrender.Node(
            name=f"plight-{i:02d}",
            light=pyrender.PointLight(color=color, intensity=intensity),
            matrix=matrix,
        )
        if scene.has_node(node):
            continue
        scene.add_node(node)

def add_lighting(scene, cam_node, color=np.ones(3), intensity=1.0):
    # from phalp.visualize.py_renderer import get_light_poses
    light_poses = get_light_poses()
    light_poses.append(np.eye(4))
    cam_pose = scene.get_pose(cam_node)
    for i, pose in enumerate(light_poses):
        matrix = cam_pose @ pose
        node = pyrender.Node(
            name=f"light-{i:02d}",
            light=pyrender.DirectionalLight(color=color, intensity=intensity),
            matrix=matrix,
        )
        if scene.has_node(node):
            continue
        scene.add_node(node)

def render_rgba_multiple(
        vertices: List[np.array],
        cam_t: List[np.array],
        rot_axis=[1,0,0],
        rot_angle=0,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0,0,0),
        render_res=[256, 256],
        focal_length=None,
        is_right=None,
    ):

    renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                          viewport_height=render_res[1],
                                          point_size=1.0)
    # material = pyrender.MetallicRoughnessMaterial(
    #     metallicFactor=0.0,
    #     alphaMode='OPAQUE',
    #     baseColorFactor=(*mesh_base_color, 1.0))

    if is_right is None:
        is_right = [1 for _ in range(len(vertices))]

    mesh_list = [pyrender.Mesh.from_trimesh(vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for vvv,ttt,sss in zip(vertices, cam_t, is_right)]

    scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    for i,mesh in enumerate(mesh_list):
        scene.add(mesh, f'mesh_{i}')

    camera_pose = np.eye(4)
    # camera_pose[:3, 3] = camera_translation
    camera_center = [render_res[0] / 2., render_res[1] / 2.]
    focal_length = focal_length if focal_length is not None else focal_length
    camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                       cx=camera_center[0], cy=camera_center[1], zfar=1e12)

    # Create camera node and add it to pyRender scene
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)
    add_point_lighting(scene, camera_node)
    add_lighting(scene, camera_node)

    light_nodes = create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    renderer.delete()

    return color

cam_view = render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

# Overlay image
input_img = image0.astype(np.float32)[:,:,::-1]/255.0
input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

input_img_overlay = input_img_overlay[:, :, ::-1]
cv2.imshow("3", input_img_overlay)
cv2.imshow("4", cam_view)
cv2.waitKey(0)

###
# mesh_base_color=(1.0, 1.0, 0.9)
# material = pyrender.MetallicRoughnessMaterial(
#             metallicFactor=0.0,
#             alphaMode='OPAQUE',
#             baseColorFactor=(*mesh_base_color, 1.0))

# import trimesh

# mesh = trimesh.Trimesh(mesh, face)
# # rot = trimesh.transformations.rotation_matrix(
# #     np.radians(180), [1, 0, 0])
# # mesh.apply_transform(rot)
# mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
# ###
# scene_bg_color=(0,0,0)
# light_nodes = create_raymond_lights()
# scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
#                                ambient_light=(0.3, 0.3, 0.3))
# scene.add(mesh, 'mesh')

# # camera_pose = np.eye(4)
# # camera_pose[:3, 3] = camera_translation
# # camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
# camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy,
#                                    cx=cx, cy=cy, zfar=zfar)
# scene.add(camera, pose=camera_pose)

# import pyrender
# renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
#                                               viewport_height=image.shape[0],
#                                               point_size=1.0)
                                              
# for node in light_nodes:
#     scene.add_node(node)

# color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
# color = color.astype(np.float32) / 255.0
# renderer.delete()

# # if return_rgba:
# #     return color

# valid_mask = (color[:, :, -1])[:, :, np.newaxis]
# side_view=False
# # if not side_view:
# output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
# # else:
# # output_img = color[:, :, :3]

# output_img = output_img.astype(np.float32)
# cv2.imshow("", output_img)
# cv2.waitKey(0)

