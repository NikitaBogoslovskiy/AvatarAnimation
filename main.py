from face_processing import landmarks
import training.dataset as ds
import training.training
import mesh_processing.visualizer as v
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pyrender
import trimesh
import time


# landmarks.track_face()
# ds.generate('files/mini_dataset.pkl', 10)
# training.training.train_model()
dataset = ds.load('files/dataset.pkl')
vis = v.Visualizer()
vis.set_surfaces(dataset['surfaces'])
for i in range(len(dataset['vertices'])):
    vis.render(dataset['vertices'][i], pause=0.07)
