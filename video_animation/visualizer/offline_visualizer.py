import numpy as np
import pyrender
import trimesh
from FLAME.flame_model import RADIAN
import cv2


class OfflineVisualizer:
    def __init__(self, save_path):
        self.surfaces = None
        self.r = None
        self.output_video = None
        self.save_path = save_path
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=[255, 255, 255], intensity=5)
        self.scene = pyrender.Scene(ambient_light=[.1, .1, .1], bg_color=[1., 1., 1.])
        light_rotation_1 = np.array([[np.cos(90 * RADIAN), np.sin(90 * RADIAN), 0, 0],
                                     [-np.sin(90 * RADIAN), np.cos(90 * RADIAN), 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
        light_rotation_2 = np.array([[np.cos(60 * RADIAN), 0, -np.sin(60 * RADIAN), 0],
                                     [0, 1, 0, 0],
                                     [0, np.sin(60 * RADIAN), np.cos(60 * RADIAN), 0],
                                     [0, 0, 0, 1]])
        light_position = np.dot(light_rotation_1, light_rotation_2)
        self.scene.add(light, pose=light_position)
        self.scene.add(camera, pose=[[1, 0, 0, 0],
                                     [0, 1, 0, -0.02],
                                     [0, 0, 1, 0.3],
                                     [0, 0, 0, 1]])
        self.object_pose = [[1, 0, 0, 0],
                            [0, np.cos(10 * RADIAN), -np.sin(10 * RADIAN), 0],
                            [0, np.sin(10 * RADIAN), np.cos(10 * RADIAN), 0],
                            [0, 0, 0, 1]]

    def set_surfaces(self, surfaces):
        self.surfaces = surfaces

    def set_resolution(self, width, height):
        self.output_video = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (width + height, height))
        self.r = pyrender.OffscreenRenderer(height, height)

    def render(self, vertices, input_frame):
        m = trimesh.Trimesh(vertices=vertices, faces=self.surfaces)
        mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
        obj = self.scene.add(mesh, pose=self.object_pose)
        output_image, _ = self.r.render(self.scene)
        self.output_video.write(np.concatenate((input_frame, output_image), axis=1))
        self.scene.remove_node(obj)

    def release(self):
        self.output_video.release()