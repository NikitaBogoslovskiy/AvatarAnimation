import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, resolution=(512, 512)):
        plt.rcParams["figure.figsize"] = [8, 8]
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 29.0)
        light = pyrender.DirectionalLight(color=[255, 247, 207], intensity=10)
        self.scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
        c = 2 ** -0.5
        self.scene.add(light, pose=np.eye(4))
        self.scene.add(camera, pose=[[1, 0, 0, 0],
                                [0, c, -c, -2],
                                [0, c, c, 2],
                                [0, 0, 0, 1]])
        self.r = pyrender.OffscreenRenderer(*resolution)
        start_image, _ = self.r.render(self.scene)
        axes = plt.subplot(111)
        self.image = axes.imshow(start_image)
        plt.ion()

    def set_surfaces(self, surfaces):
        self.surfaces = surfaces

    def render(self, vertices, pause=None):
        m = trimesh.Trimesh(vertices=vertices, faces=self.surfaces)
        mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
        obj = self.scene.add(mesh, pose=[[1, 0, 0, 0],
                              [0, 0.78, -0.78, -0.02],
                              [0, 0.78, 0.78, 0],
                              [0, 0, 0, 1]])
        color, _ = self.r.render(self.scene)
        self.image.set_data(color)
        if pause is None:
            plt.waitforbuttonpress()
        else:
            plt.pause(pause)
        self.scene.remove_node(obj)

