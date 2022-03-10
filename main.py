from face import landmarks
import model.dataset as ds
from model.wrapper import ModelWrapper
import mesh.visualizer as v
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pyrender
import trimesh
import time
from animation import Animation


# landmarks.track_face()
# ds.generate('files/dataset.pkl', 60000, expr_bound=1.8, pose_bound=np.pi/13.5)
# model.model.train_model()
# dataset = ds.load('files/dataset.pkl')
# vis = v.Visualizer()
# vis.set_surfaces(dataset['surfaces'])
# for i in range(len(dataset['vertices'])):
#     vis.render(dataset['vertices'][i], pause=0.07)

# det = landmarks.Detector()
# det.load_image('files/face_example.jpg')
# det.detect_face()
# det.detect_landmarks()
# model = ModelWrapper()
# model.load_dataset()
# model.update_dataset()
# model.gen_release_data()
# model.save_release_data()
# model.load_model()
# model.train_and_test()
# model.save_model()
a = Animation()
a.capture_neutral_face()
a.animate_mesh()
a.stop()
# landmarks.draw_it(np.array([[0, 0, 0]]))
