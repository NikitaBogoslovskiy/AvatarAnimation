import pickle
import numpy as np


def generate(path, num_samples, expr_bound=2.3, pose_bound=(np.pi/6)):
    with open('files/generic_model.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    shape_expr_basis = data['shapedirs']
    pose_basis = data['posedirs']
    neutral_face = data['v_template']
    surfaces = data['f']

    num_vertices = len(neutral_face)
    num_shape_params = 300
    num_expr_params = len(shape_expr_basis[0][0]) - num_shape_params
    num_pose_params = len(pose_basis[0][0])

    dataset = dict()
    dataset['vertices'] = np.zeros((num_samples, num_vertices, 3))
    dataset['shape_params'] = np.zeros((num_samples, num_shape_params, 1))
    dataset['expr_params'] = np.zeros((num_samples, num_expr_params, 1))
    dataset['pose_params'] = np.zeros((num_samples, num_pose_params, 1))
    dataset['shape_expr_basis'] = np.array(shape_expr_basis)
    dataset['pose_basis'] = np.array(pose_basis)
    dataset['neutral_face'] = np.array(neutral_face)
    dataset['surfaces'] = np.array(surfaces, dtype='int32')

    print('Generating dataset...')
    for i in range(num_samples):
        expr_coefs1 = np.zeros((num_shape_params + num_expr_params, 1))
        expr_coefs1[num_shape_params:] = np.random.rand(num_expr_params, 1) * expr_bound * 2 - expr_bound
        expr_coefs2 = np.random.rand(num_pose_params, 1) * pose_bound
        shape_expr = np.dot(shape_expr_basis.reshape((num_vertices * 3, num_shape_params + num_expr_params)),
                            expr_coefs1)
        pose = np.dot(pose_basis.reshape((num_vertices * 3, num_pose_params)), expr_coefs2)
        vertices = shape_expr.reshape(-1, 3) + pose.reshape(-1, 3) + neutral_face

        dataset['vertices'][i] = vertices
        dataset['shape_params'][i] = expr_coefs1[:num_shape_params]
        dataset['expr_params'][i] = expr_coefs1[num_shape_params:]
        dataset['pose_params'][i] = expr_coefs2

        if (i + 1) % 1000 == 0:
            print(f'{i + 1}/{num_samples}')

    print(f'Saving dataset to "{path}"...')
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print('Dataset has been created and saved successfully!')


def load(path):
    print(f'Uploading dataset...')
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    print(f'Dataset has been uploaded successfully')
    return dataset


if __name__ == '__main__':
    pass
