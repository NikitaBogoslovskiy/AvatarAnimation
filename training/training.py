import torch
from ikmodel import IKModel
import dataset
import face_processing.landmarks as landmarks


# Load old and create new dataset
dataset_path = '../files/mini_dataset.pkl'
data = dataset.load(dataset_path)
landmarks_path = '../files/flame_static_embedding.pkl'
indices, coordinates = landmarks.load_mesh_landmarks(landmarks_path)
lm_coordinates = landmarks.meshes_to_landmarks(data['vertices'], data['surfaces'], indices, coordinates)
les, res, nms = landmarks.divide_landmarks(lm_coordinates)

new_data = dict()
for key, value in data.items():
    new_data[key] = torch.Tensor(value)
new_data['indices'] = torch.Tensor(indices)
new_data['coordinates'] = torch.Tensor(coordinates)
new_data['landmark_coordinates'] = torch.Tensor(lm_coordinates)
new_data['left_eyes'] = torch.Tensor(les)
new_data['right_eyes'] = torch.Tensor(res)
new_data['noses_mouths'] = torch.Tensor(nms)


if __name__ == '__main__':
    pass
