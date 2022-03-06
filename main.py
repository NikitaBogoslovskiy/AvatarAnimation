from face_processing import landmarks
import training.dataset as ds
import training.training
from mesh_processing.visualizer import render_mesh


# landmarks.track_face()
# ds.generate('files/mini_dataset.pkl', 10)
# training.training.train_model()
dataset = ds.load('files/mini_dataset.pkl')
render_mesh(dataset['vertices'][9], dataset['surfaces'], path='output/rendered_mesh.jpg')
