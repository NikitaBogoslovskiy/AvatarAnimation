import torch
from training.ikmodel import IKModel
import training.dataset as dataset
import face_processing.landmarks as landmarks
import numpy as np
from training.funcs import init_weights, draw_mesh


def train_model():
    # Load dataset
    dataset_path = 'files/dataset.pkl'
    print('Loading dataset... ', end=' ')
    data = dataset.load(dataset_path)
    print('Success')
    landmarks_path = 'files/flame_static_embedding.pkl'
    indices, coordinates = landmarks.load_mesh_landmarks(landmarks_path)
    indices = indices.astype(dtype='int64')
    lm_coordinates = landmarks.meshes_to_landmarks_numpy(data['vertices'], data['surfaces'], indices, coordinates)
    les, res, nms = landmarks.divide_landmarks(lm_coordinates)

    # Update dataset
    new_data = dict()
    for key, value in data.items():
        new_data[key] = torch.Tensor(value)
    new_data['surfaces'] = new_data['surfaces'].type(torch.long)
    new_data['indices'] = torch.Tensor(indices).type(torch.int64)
    new_data['coordinates'] = torch.Tensor(coordinates)
    new_data['landmark_coordinates'] = torch.Tensor(lm_coordinates)
    new_data['left_eyes'] = torch.Tensor(les)
    new_data['right_eyes'] = torch.Tensor(res)
    new_data['noses_mouths'] = torch.Tensor(nms)

    # Prepare to train
    num_samples = 60000
    num_train = 58800
    batch_size = 400
    shape_expr_basis = torch.transpose(new_data['shape_expr_basis'].view(15069, 400), 0, 1)
    pose_basis = torch.transpose(new_data['pose_basis'].view(15069, 36), 0, 1)
    neutral_face = new_data['neutral_face'][None, :].repeat(batch_size, 1, 1)

    model = IKModel()
    model.apply(init_weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    decayRate = 0.98
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=decayRate)
    noise_coef = 1e-4

    # Training
    for i in range(0, num_train, batch_size):
        noise = torch.rand((batch_size, 51, 3)) * noise_coef
        left_eyes = new_data['left_eyes'][i:i + batch_size] + noise[:, :11]
        right_eyes = new_data['right_eyes'][i:i + batch_size] + noise[:, 11:22]
        noses_mouths = new_data['noses_mouths'][i:i + batch_size] + noise[:, 22:51]
        landmark_coordinates = new_data['landmark_coordinates'][i:i + batch_size] + noise
        result = model(left_eyes, right_eyes, noses_mouths)
        expr_params = result[:, :100]
        pose_params = result[:, 100:]
        shape_expr_params = torch.cat([new_data['shape_params'][i:i+batch_size, :, 0], expr_params], 1)
        shape_expr = torch.mm(shape_expr_params, shape_expr_basis).view(-1, 5023, 3)
        pose = torch.mm(pose_params, pose_basis).view(-1, 5023, 3)
        vertices = neutral_face + shape_expr + pose
        new_lm_coordinates = landmarks.meshes_to_landmarks_torch(vertices, new_data['surfaces'], new_data['indices'], new_data['coordinates'])
        loss = torch.sum(torch.abs(new_lm_coordinates - landmark_coordinates)) / batch_size
        loss.backward()
        opt.step()
        if i % 20000 == 0:
            lr_scheduler.step()
        opt.zero_grad()
        print(f'{i+batch_size}/{num_samples}: loss = {loss}')

    # Test
    loss_history = []
    num_examples = 5
    old_faces_indices = np.random.choice(np.array(range(num_train, num_train + batch_size)), num_examples)
    new_faces_indices = old_faces_indices - num_train
    old_faces = None
    new_faces = None
    for i in range(num_train, num_samples, batch_size):
        left_eyes = new_data['left_eyes'][i:i + batch_size]
        right_eyes = new_data['right_eyes'][i:i + batch_size]
        noses_mouths = new_data['noses_mouths'][i:i + batch_size]
        landmark_coordinates = new_data['landmark_coordinates'][i:i + batch_size]
        result = model(left_eyes, right_eyes, noses_mouths)
        expr_params = result[:, :100]
        pose_params = result[:, 100:]
        shape_expr_params = torch.cat([new_data['shape_params'][i:i+batch_size, :, 0], expr_params], 1)
        shape_expr = torch.mm(shape_expr_params, shape_expr_basis).view(-1, 5023, 3)
        pose = torch.mm(pose_params, pose_basis).view(-1, 5023, 3)
        vertices = neutral_face + shape_expr + pose
        new_lm_coordinates = landmarks.meshes_to_landmarks_torch(vertices, new_data['surfaces'], new_data['indices'], new_data['coordinates'])
        loss = torch.sum(torch.abs(new_lm_coordinates - landmark_coordinates)) / batch_size
        loss_history.append(loss)
        if i == num_train:
            old_faces = data['vertices'][old_faces_indices]
            new_faces = vertices[new_faces_indices].detach().numpy()

    print(f'Eventual loss = {sum(loss_history) / len(loss_history)}')

    # Save model
    model_path = 'files/IKModel.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model has been saved to "{model_path}"')

    # Examples
    num_examples = 5
    for i in range(num_examples):
        draw_mesh(old_faces[i], data['surfaces'], f'output/old_mesh{i+1}.obj')
        draw_mesh(new_faces[i], data['surfaces'], f'output/new_mesh{i+1}.obj')
    print('In files directory you can find illustrations of training results')


if __name__ == '__main__':
    pass
