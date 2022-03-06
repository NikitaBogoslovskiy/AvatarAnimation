import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt


def render_mesh(vertices, surfaces, path):
    m = trimesh.Trimesh(vertices=vertices, faces=surfaces)
    mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 29.0)
    light = pyrender.DirectionalLight(color=[255, 247, 207], intensity=10)
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
    c = 2 ** -0.5
    scene.add(mesh, pose=[[1, 0, 0, 0],
                          [0, 0.78, -0.78, -0.02],
                          [0, 0.78, 0.78, 0],
                          [0, 0, 0, 1]])
    scene.add(light, pose=np.eye(4))
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, c, -c, -2],
                            [0, c, c, 2],
                            [0, 0, 0, 1]])
    r = pyrender.OffscreenRenderer(1024, 1024)
    color, _ = r.render(scene)
    plt.figure(figsize=(8, 8))
    plt.imshow(color)
    plt.imsave(path, color)
