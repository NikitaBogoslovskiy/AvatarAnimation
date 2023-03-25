import numpy as np
import torch
from FLAME import FLAME
import pyrender
import trimesh
from config import get_config


RADIAN = np.pi / 180.0


class FlameModel:
    def __init__(self, config, cuda=True):
        self.config = config
        self.flamelayer = FLAME(config)
        self.cuda = cuda
        if self.cuda:
            self.flamelayer.cuda()

    def generate(self, shape, pose, expression, neck=None, eye=None):
        if shape.is_cuda != self.cuda or pose.is_cuda != self.cuda or expression.is_cuda != self.cuda:
            raise Exception("Cuda settings of class and function parameters are different")
        if neck is None and eye is None:
            vertices, landmarks = self.flamelayer(shape, expression, pose)
        else:
            vertices, landmarks = self.flamelayer(shape, expression, pose, neck, eye)
        return vertices, landmarks

    def draw(self, vertices, landmarks=None):
        processed_vertices = vertices.detach().cpu().numpy().squeeze()
        if landmarks is not None:
            processed_landmarks = landmarks.detach().cpu().numpy().squeeze()
        vertex_colors = np.ones([processed_vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 1.0]
        tri_mesh = trimesh.Trimesh(processed_vertices, self.flamelayer.faces,
                                   vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        if landmarks is not None:
            sm = trimesh.creation.uv_sphere(radius=0.002)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(processed_landmarks), 1, 1))
            tfs[:, :3, 3] = processed_landmarks
            processed_landmarks_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(processed_landmarks_pcl)
        pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    fm = FlameModel(get_config())
    shape_params = torch.zeros(1, 100).cuda()
    pose_params_numpy = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()
    expression_params = torch.zeros(1, 50, dtype=torch.float32).cuda()
    v, l = fm.generate(shape_params, pose_params, expression_params)
    fm.draw(v, l)
