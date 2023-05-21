import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from FLAME.flame_model import RADIAN


class OnlineVisualizer:
    def __init__(self, resolution=(512, 512)):
        self.surfaces = None
        plt.rcParams["figure.figsize"] = [8, 8]
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
        self.r = pyrender.OffscreenRenderer(*resolution)
        start_image, _ = self.r.render(self.scene)
        plt.suptitle('', fontsize=16)
        axes = plt.subplot(111)
        self.image = axes.imshow(start_image)
        plt.ion()

    def set_surfaces(self, surfaces):
        self.surfaces = surfaces

    def render(self, vertices, pause=None):
        m = trimesh.Trimesh(vertices=vertices, faces=self.surfaces)
        mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
        obj = self.scene.add(mesh, pose=self.object_pose)
        color, _ = self.r.render(self.scene)
        self.image.set_data(color)
        if pause is None:
            plt.waitforbuttonpress()
        else:
            plt.pause(pause)
        self.scene.remove_node(obj)

    def release(self):
        plt.ioff()
        plt.close()
