import torch
from model.ikmodel import IKModel
import model.dataset as dataset
import face.landmarks as landmarks
import numpy as np
from model.funcs import init_weights, draw_mesh
import pickle
import os


class ModelWrapper:
    def __init__(self):
        self.model = None

    def load_dataset(self, dataset_path='files/dataset.pkl', landmarks_path='files/flame_static_embedding.pkl'):
        self.data = dataset.load(dataset_path)
        indices, self.coordinates = landmarks.load_mesh_landmarks(landmarks_path)
        self.indices = indices.astype(dtype='int64')
        self.lm_coordinates = landmarks.meshes_to_landmarks_numpy(self.data['vertices'], self.data['surfaces'], self.indices, self.coordinates)
        les, res, nms = landmarks.divide_landmarks(self.lm_coordinates)
        self.left_eye_inputs = les[:, :, :2]
        self.right_eye_inputs = res[:, :, :2]
        self.nose_mouth_inputs = nms[:, :, :2]
        nlms = landmarks.meshes_to_landmarks_numpy(self.data['neutral_face'][None, :], self.data['surfaces'], self.indices, self.coordinates)
        l, r, nm = landmarks.divide_landmarks(nlms)
        self.left_eye_neutral = l[:, :, :2]
        self.right_eye_neutral = r[:, :, :2]
        self.nose_mouth_neutral = nm[:, :, :2]

    def update_dataset(self):
        self.new_data = dict()
        for key, value in self.data.items():
            self.new_data[key] = torch.Tensor(value)
        self.new_data['surfaces'] = self.new_data['surfaces'].type(torch.long)
        self.new_data['indices'] = torch.Tensor(self.indices).type(torch.int64)
        self.new_data['coordinates'] = torch.Tensor(self.coordinates)
        self.new_data['landmark_coordinates'] = torch.Tensor(self.lm_coordinates)
        self.new_data['left_eyes'] = torch.Tensor(self.left_eye_inputs)
        self.new_data['right_eyes'] = torch.Tensor(self.right_eye_inputs)
        self.new_data['noses_mouths'] = torch.Tensor(self.nose_mouth_inputs)

    def gen_release_data(self):
        self.release_data = dict()
        self.release_data['left_eye'] = torch.Tensor(self.left_eye_neutral)
        self.release_data['right_eye'] = torch.Tensor(self.right_eye_neutral)
        self.release_data['nose_mouth'] = torch.Tensor(self.nose_mouth_neutral)
        self.release_data['neutral_face'] = self.new_data['neutral_face']
        self.release_data['shape_params'] = self.new_data['shape_params'][0]
        self.release_data['shape_expr_basis'] = torch.transpose(self.new_data['shape_expr_basis'].view(15069, 400), 0, 1)
        self.release_data['pose_basis'] = torch.transpose(self.new_data['pose_basis'].view(15069, 36), 0, 1)
        self.release_data['surfaces'] = self.new_data['surfaces']

    def save_release_data(self, path='files/release_data.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.release_data, f)
        print(f'Release data has been saved successfully to the path {path}')

    def load_release_data(self, path='files/release_data.pkl'):
        with open(path, 'rb') as f:
            self.release_data = pickle.load(f)
        print('Release data has been successfully loaded')

    def train_and_test(self):
        # Prepare to train
        num_samples = 60000
        num_train = 58800
        batch_size = 400
        shape_expr_basis = torch.transpose(self.new_data['shape_expr_basis'].view(15069, 400), 0, 1)
        pose_basis = torch.transpose(self.new_data['pose_basis'].view(15069, 36), 0, 1)
        neutral_face = self.new_data['neutral_face'][None, :].repeat(batch_size, 1, 1)

        self.model = IKModel()
        self.model.apply(init_weights)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        decayRate = 0.98
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=decayRate)
        noise_coef = 1e-4

        # Training
        for i in range(0, num_train, batch_size):
            noise = torch.rand((batch_size, 51, 3)) * noise_coef
            noise[:, :, 2] = 0
            left_eyes = self.new_data['left_eyes'][i:i + batch_size] + noise[:, :11, :2]
            right_eyes = self.new_data['right_eyes'][i:i + batch_size] + noise[:, 11:22, :2]
            noses_mouths = self.new_data['noses_mouths'][i:i + batch_size] + noise[:, 22:51, :2]
            landmark_coordinates = self.new_data['landmark_coordinates'][i:i + batch_size] + noise
            result = self.model(left_eyes, right_eyes, noses_mouths)
            expr_params = result[:, :100]
            pose_params = result[:, 100:]
            shape_expr_params = torch.cat([self.new_data['shape_params'][i:i + batch_size, :, 0], expr_params], 1)
            shape_expr = torch.mm(shape_expr_params, shape_expr_basis).view(-1, 5023, 3)
            pose = torch.mm(pose_params, pose_basis).view(-1, 5023, 3)
            vertices = neutral_face + shape_expr + pose
            new_lm_coordinates = landmarks.meshes_to_landmarks_torch(vertices, self.new_data['surfaces'],
                                                                     self.new_data['indices'], self.new_data['coordinates'])
            loss = torch.sum(torch.abs(new_lm_coordinates - landmark_coordinates)) / batch_size
            loss.backward()
            opt.step()
            if i % 20000 == 0:
                lr_scheduler.step()
            opt.zero_grad()
            print(f'{i + batch_size}/{num_samples}: loss = {loss}')

        # Test
        loss_history = []
        num_examples = 3
        old_faces_indices = np.random.choice(np.array(range(num_train, num_train + batch_size)), num_examples)
        new_faces_indices = old_faces_indices - num_train
        old_faces = None
        new_faces = None
        for i in range(num_train, num_samples, batch_size):
            left_eyes = self.new_data['left_eyes'][i:i + batch_size]
            right_eyes = self.new_data['right_eyes'][i:i + batch_size]
            noses_mouths = self.new_data['noses_mouths'][i:i + batch_size]
            landmark_coordinates = self.new_data['landmark_coordinates'][i:i + batch_size]
            result = self.model(left_eyes, right_eyes, noses_mouths)
            expr_params = result[:, :100]
            pose_params = result[:, 100:]
            shape_expr_params = torch.cat([self.new_data['shape_params'][i:i + batch_size, :, 0], expr_params], 1)
            shape_expr = torch.mm(shape_expr_params, shape_expr_basis).view(-1, 5023, 3)
            pose = torch.mm(pose_params, pose_basis).view(-1, 5023, 3)
            vertices = neutral_face + shape_expr + pose
            new_lm_coordinates = landmarks.meshes_to_landmarks_torch(vertices, self.new_data['surfaces'],
                                                                     self.new_data['indices'], self.new_data['coordinates'])
            loss = torch.sum(torch.abs(new_lm_coordinates - landmark_coordinates)) / batch_size
            loss_history.append(loss)
            if i == num_train:
                old_faces = self.data['vertices'][old_faces_indices]
                new_faces = vertices[new_faces_indices].detach().numpy()

        print(f'Eventual loss = {sum(loss_history) / len(loss_history)}')

        for i in range(num_examples):
            draw_mesh(old_faces[i], self.data['surfaces'], f'output/old_mesh{i + 1}.obj')
            draw_mesh(new_faces[i], self.data['surfaces'], f'output/new_mesh{i + 1}.obj')
        print('In files directory you can find illustrations of model results')

    def save_model(self, path='files/IKModel.pt'):
        torch.save(self.model.state_dict(), path)
        print(f'Model has been saved to "{path}"')

    def load_model(self, path='files/IKModel.pt'):
        self.model = IKModel()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f'Model has been successfully loaded')

    def execute(self, left_eyes, right_eyes, noses_mouths):
        result = self.model(left_eyes, right_eyes, noses_mouths)
        expr_params = torch.clip(result[:, :100], min=-1.72, max=1.72)
        pose_params = torch.clip(result[:, 100:], min=-np.pi/14.6, max=np.pi/14.6)
        batch_size = len(expr_params)
        shape_params = self.release_data['shape_params'][None, :, 0].repeat(batch_size, 1)
        neutral_face = self.release_data['neutral_face'][None, :].repeat(batch_size, 1, 1)
        shape_expr_params = torch.cat([shape_params, expr_params], 1)
        shape_expr = torch.mm(shape_expr_params, self.release_data['shape_expr_basis']).view(-1, 5023, 3)
        pose = torch.mm(pose_params, self.release_data['pose_basis']).view(-1, 5023, 3)
        vertices = neutral_face + shape_expr + pose
        return vertices.detach().numpy()


if __name__ == '__main__':
    pass
