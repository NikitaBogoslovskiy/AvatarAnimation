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
        light_rotation_1 = np.array([[np.cos(45 * RADIAN), np.sin(45 * RADIAN), 0, 0],
                                     [-np.sin(45 * RADIAN), np.cos(45 * RADIAN), 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
        light_rotation_2 = np.array([[np.cos(30 * RADIAN), 0, -np.sin(30 * RADIAN), 0],
                                     [0, 1, 0, 0],
                                     [np.sin(30 * RADIAN), 0, np.cos(30 * RADIAN), 0],
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

    def init_settings(self, animation_resolution, input_resolution, frame_rate):
        if input_resolution is None:
            final_resolution = animation_resolution
        else:
            if animation_resolution[1] != input_resolution[1]:
                raise Exception("Height of input frame and height of generated animation must be the same")
            final_resolution = (animation_resolution[0] + input_resolution[0], animation_resolution[1])
        self.output_video = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, final_resolution)
        self.r = pyrender.OffscreenRenderer(*animation_resolution)

    def set_resolution(self, width, height, frame_rate=30):
        self.output_video = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width + height, height))
        self.r = pyrender.OffscreenRenderer(height, height)

    def render(self, vertices, input_frame=None):
        m = trimesh.Trimesh(vertices=vertices, faces=self.surfaces, vertex_colors=[0.5, 0.5, 0.5])
        mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
        obj = self.scene.add(mesh, pose=self.object_pose)
        output_image, _ = self.r.render(self.scene)
        final_frame = output_image if input_frame is None else np.concatenate((input_frame, output_image), axis=1)
        self.output_video.write(final_frame)
        self.scene.remove_node(obj)

    def release(self):
        self.output_video.release()
