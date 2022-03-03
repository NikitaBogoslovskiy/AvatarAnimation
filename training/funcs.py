import torch
from torch import nn
import openmesh as om


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def draw_mesh(vertices, surfaces, path):
    m = om.TriMesh()
    points = []
    for i in range(vertices.shape[0]):
        points.append(m.add_vertex(vertices[i]))
    for j in range(surfaces.shape[0]):
        m.add_face(points[surfaces[j, 0]], points[surfaces[j, 1]], points[surfaces[j, 2]])
    om.write_mesh(mesh=m, filename=path)


if __name__ == '__main__':
    pass
