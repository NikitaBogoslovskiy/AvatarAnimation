import torch
from FLAME.flame_model import FlameModel
import json
from FLAME.config import get_config
from progress.bar import Bar
from FLAME.flame_model import RADIAN
from utils.landmarks import divide_landmarks

JAW_ARTICULATION_PROBABILITY = 0.3


class DatasetParams:
    def __init__(self, save_folder, num_samples, expr_min, expr_max, jaw_min, jaw_max):
        self.save_folder = save_folder
        self.num_samples = num_samples
        self.expr_min = expr_min
        self.expr_max = expr_max
        self.jaw_min = jaw_min
        self.jaw_max = jaw_max


class Dataset:
    @staticmethod
    def save(path, vertices, landmarks):
        data_item = dict()
        data_item["vertices"] = vertices
        data_item["landmarks"] = landmarks
        with open(path, "w") as f:
            f.write(json.dumps(data_item))

    @staticmethod
    def upload(path):
        with open(path, "r") as f:
            data_item = json.loads(f.read())
        if "vertices" not in data_item or "landmarks" not in data_item:
            raise Exception("Wrong file. Must contain 'vertices' and 'landmarks' fields")
        return data_item["vertices"], data_item["landmarks"]

    @staticmethod
    def generate(params, cuda=True, batch_size=8):
        model = FlameModel(get_config(batch_size), cuda)
        num_batches = params.num_samples // batch_size
        num_items = num_batches * batch_size
        bar = Bar('Generated data items', max=num_items, check_tty=False)
        shape = torch.zeros(num_items, model.config.shape_params)
        pose = torch.zeros(num_items, model.config.pose_params)
        pose[:, 3:] = torch.rand(num_items, 3) * (params.jaw_max - params.jaw_min).repeat(num_items, 1) \
                             + params.jaw_min.repeat(num_items, 1)
        pose[:, 3:] *= (torch.rand(num_items, 1) < JAW_ARTICULATION_PROBABILITY).repeat(1, 3)
        expr = torch.rand(num_items, model.config.expression_params) * (params.expr_max - params.expr_min)\
                      + params.expr_min
        shape[0, :] = 0
        pose[0, :] = 0
        expr[0, :] = 0
        if cuda:
            shape = shape.cuda()
            pose = pose.cuda()
            expr = expr.cuda()
        global_item_idx = 0
        print("Started generating dataset")
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            vertices, landmarks = model.generate(shape[start_idx:end_idx], pose[start_idx:end_idx], expr[start_idx:end_idx])
            processed_vertices = vertices.detach().cpu().numpy().squeeze().tolist()
            processed_landmarks = landmarks.detach().cpu().numpy().squeeze().tolist()
            for local_item_idx in range(batch_size):
                if global_item_idx == 0:
                    Dataset.save(f"{params.save_folder}/neutral.json",
                                 processed_vertices[local_item_idx], processed_landmarks[local_item_idx])
                else:
                    Dataset.save(f"{params.save_folder}/{global_item_idx}.json",
                                 processed_vertices[local_item_idx], processed_landmarks[local_item_idx])
                # le, re, nm = divide_landmarks(landmarks[local_item_idx])
                # model.draw(vertices[local_item_idx], landmarks[local_item_idx])
                # model.draw_with_divided_landmarks(vertices[local_item_idx], le, re, nm)
                bar.next()
                global_item_idx += 1
        bar.finish()
        print(f"Successfully finished. Dataset has been saved to {params.save_folder}")


if __name__ == "__main__":
    p = DatasetParams(save_folder="train_data",
                      num_samples=10000,
                      expr_min=-2.5,
                      expr_max=2.5,
                      jaw_min=torch.Tensor([0.0 * RADIAN, -2.0 * RADIAN, -3.0 * RADIAN]),
                      jaw_max=torch.Tensor([25.0 * RADIAN, 3.0 * RADIAN, 4.0 * RADIAN]))
    Dataset.generate(p)
