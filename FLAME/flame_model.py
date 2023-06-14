import numpy as np
from FLAME.FLAME import FLAME as flame
import pyrender
import trimesh
from trimesh.exchange.obj import export_obj

RADIAN = np.pi / 180.0


class FlameModel:
    def __init__(self, configuration, cuda=True):
        self.config = configuration
        self.flamelayer = flame(self.config)
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

    def _save(self, vertices, filepath):
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 1.0]
        tri_mesh = trimesh.Trimesh(vertices, self.flamelayer.faces)
        str_obj = trimesh.exchange.obj.export_obj(tri_mesh)
        with open(filepath, "w") as f:
            f.write(str_obj)

    def _draw(self, vertices, landmarks=None):
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
            # processed_landmarks[:, 2] = 0
            tfs[:, :3, 3] = processed_landmarks
            processed_landmarks_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(processed_landmarks_pcl)
        pyrender.Viewer(scene, use_raymond_lighting=True)

    def _draw_with_divided_landmarks(self, vertices, left_eye, right_eye, nose_mouth):
        processed_vertices = vertices.detach().cpu().numpy().squeeze()
        processed_left_eye = left_eye.detach().cpu().numpy().squeeze()
        processed_right_eye = right_eye.detach().cpu().numpy().squeeze()
        processed_nose_mouth = nose_mouth.detach().cpu().numpy().squeeze()
        vertex_colors = np.ones([processed_vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.1]
        tri_mesh = trimesh.Trimesh(processed_vertices, self.flamelayer.faces,
                                   vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        landmark_parts_coordinates = [processed_left_eye, processed_right_eye, processed_nose_mouth]
        landmark_parts_colors = [[0.9, 0.1, 0.1, 1.0], [0.1, 0.9, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0]]
        for i in range(3):
            sm = trimesh.creation.uv_sphere(radius=0.002)
            sm.visual.vertex_colors = landmark_parts_colors[i]
            tfs = np.tile(np.eye(4), (len(landmark_parts_coordinates[i]), 1, 1))
            landmark_parts_coordinates[i][:, 2] = 0
            tfs[:, :3, 3] = landmark_parts_coordinates[i]
            landmarks_part_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(landmarks_part_pcl)
        pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    pass
