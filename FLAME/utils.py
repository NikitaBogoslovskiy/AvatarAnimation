from config.paths import PROJECT_DIR
import torch
import pickle


class Masks:
    def __init__(self,
                 forehead,
                 eye_region,
                 nose):
        self.forehead = forehead
        self.eye_region = eye_region
        self.nose = nose


def upload_face_mask(path=f"{PROJECT_DIR}/FLAME/model/FLAME_masks.pkl"):
    with open(path, 'rb') as f:
        face_mask = torch.Tensor(pickle.load(f, encoding='latin1')['face']).type(torch.IntTensor)
    return face_mask


def upload_lips_mask(path=f"{PROJECT_DIR}/FLAME/model/FLAME_masks.pkl"):
    with open(path, 'rb') as f:
        lips_mask = torch.Tensor(pickle.load(f, encoding='latin1')['lips']).type(torch.IntTensor)
    return lips_mask


def upload_masks(path=f"{PROJECT_DIR}/FLAME/model/FLAME_masks.pkl"):
    with open(path, 'rb') as f:
        masks_dict = pickle.load(f, encoding='latin1')
        masks = Masks(forehead=masks_dict["forehead"],
                      eye_region=masks_dict["eye_region"],
                      nose=masks_dict["nose"])
    return masks
