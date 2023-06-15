from config.paths import PROJECT_DIR
import torch
from FLAME.flame_model import FlameModel
import json
from FLAME.config import get_config
from progress.bar import Bar
from FLAME.flame_model import RADIAN
from FLAME.utils import upload_masks
from video_animation.visualizer.online_visualizer import OnlineVisualizer

JAW_ARTICULATION_PROBABILITY = 0.15


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
    def save(path, **kwargs):
        data_item = dict()
        for name, data in kwargs.items():
            data_item[name] = data
        with open(path, "w") as f:
            f.write(json.dumps(data_item))

    @staticmethod
    def save_neutral(path, vertices, landmarks):
        data_item = dict()
        data_item["vertices"] = vertices
        data_item["landmarks"] = landmarks
        with open(path, "w") as f:
            f.write(json.dumps(data_item))

    @staticmethod
    def upload(path):
        with open(path, "r") as f:
            data_item = json.loads(f.read())
        return data_item

    @staticmethod
    def upload_neutral(path):
        with open(path, "r") as f:
            data_item = json.loads(f.read())
        if "vertices" not in data_item or "landmarks" not in data_item:
            raise Exception("Wrong file. Must contain 'vertices' and 'landmarks' fields")
        return data_item["vertices"], data_item["landmarks"]

    @staticmethod
    def generate(params, cuda=True, batch_size=500):
        model = FlameModel(get_config(batch_size), cuda)
        num_batches = params.num_samples // batch_size
        num_items = num_batches * batch_size
        masks = upload_masks()
        bar = Bar('Generated data items', max=num_items, check_tty=False)
        shape = torch.zeros(num_items, model.config.shape_params)
        pose = torch.zeros(num_items, model.config.pose_params)
        pose[:, 3] = torch.clip(torch.abs(torch.normal(mean=0.0 * RADIAN, std=4 * RADIAN, size=(num_items,))), min=params.jaw_min, max=params.jaw_max)
        pose[:, 3] *= (torch.rand(num_items) < JAW_ARTICULATION_PROBABILITY)
        expr = torch.rand(num_items, model.config.expression_params) * (params.expr_max - params.expr_min) \
               + params.expr_min
        shape[0, :] = 0
        pose[0, :] = 0
        expr[0, :] = 0
        if cuda:
            shape = shape.cuda()
            pose = pose.cuda()
            expr = expr.cuda()
        global_item_idx = 1

        v = OnlineVisualizer()
        v.set_surfaces(model.flamelayer.faces)
        print("Started generating dataset")
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            vertices, landmarks = model.generate(shape[start_idx:end_idx], pose[start_idx:end_idx],
                                                 expr[start_idx:end_idx])
            all_vertices = vertices.detach().cpu()
            landmarks = landmarks.detach().cpu().numpy().squeeze().tolist()
            if batch_idx == 0:
                Dataset.save_neutral(f"{params.save_folder}/neutral.json",
                                     all_vertices.numpy()[0].squeeze().tolist(), landmarks[0])
                forehead = all_vertices[:, masks.forehead].numpy().squeeze().tolist()
                left_eye_region = all_vertices[:, masks.left_eye_region].numpy().squeeze().tolist()
                right_eye_region = all_vertices[:, masks.right_eye_region].numpy().squeeze().tolist()
                nose = all_vertices[:, masks.nose].numpy().squeeze().tolist()
                lips = all_vertices[:, masks.lips].numpy().squeeze().tolist()
                # for local_item_idx in range(1, 50):
                #     v.save(all_vertices[local_item_idx].numpy().squeeze(), f"{PROJECT_DIR}/video_animation/dataset/imgs/{global_item_idx}_{torch.min(expr[local_item_idx])}_{torch.max(expr[local_item_idx])}.jpg")
                #     global_item_idx += 1
                for local_item_idx in range(1, batch_size):
                    Dataset.save(f"{params.save_folder}/{global_item_idx}.json",
                                 forehead=forehead[local_item_idx],
                                 left_eye_region=left_eye_region[local_item_idx],
                                 right_eye_region=right_eye_region[local_item_idx],
                                 nose=nose[local_item_idx],
                                 lips=lips[local_item_idx],
                                 landmarks=landmarks[local_item_idx])
                    bar.next()
                    global_item_idx += 1
                continue
            forehead = all_vertices[:, masks.forehead].numpy().squeeze().tolist()
            left_eye_region = all_vertices[:, masks.left_eye_region].numpy().squeeze().tolist()
            right_eye_region = all_vertices[:, masks.right_eye_region].numpy().squeeze().tolist()
            nose = all_vertices[:, masks.nose].numpy().squeeze().tolist()
            lips = all_vertices[:, masks.lips].numpy().squeeze().tolist()
            for local_item_idx in range(batch_size):
                Dataset.save(f"{params.save_folder}/{global_item_idx}.json",
                             forehead=forehead[local_item_idx],
                             left_eye_region=left_eye_region[local_item_idx],
                             right_eye_region=right_eye_region[local_item_idx],
                             nose=nose[local_item_idx],
                             lips=lips[local_item_idx],
                             landmarks=landmarks[local_item_idx])
                bar.next()
                global_item_idx += 1
        bar.finish()
        print(f"Successfully finished. Dataset has been saved to {params.save_folder}")


if __name__ == "__main__":
    p = DatasetParams(save_folder=f"{PROJECT_DIR}/video_animation/dataset/train_data",
                      num_samples=100000,
                      expr_min=-2.5,
                      expr_max=2.75,
                      jaw_min=0.0 * RADIAN,
                      jaw_max=14 * RADIAN)
    Dataset.generate(p, batch_size=500)
