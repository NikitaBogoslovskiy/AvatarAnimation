from config.paths import PROJECT_DIR
import torch
import pickle


class UpperMasks:
    def __init__(self,
                 forehead,
                 eye_region,
                 nose):
        self.forehead = forehead
        self.eye_region = eye_region
        self.nose = nose


class Masks:
    def __init__(self,
                 forehead,
                 left_eye_region,
                 right_eye_region,
                 nose,
                 lips):
        self.forehead = forehead
        self.left_eye_region = left_eye_region
        self.right_eye_region = right_eye_region
        self.nose = nose
        self.lips = lips

    def to_cuda(self):
        self.forehead = self.forehead.cuda()
        self.left_eye_region = self.left_eye_region.cuda()
        self.right_eye_region = self.right_eye_region.cuda()
        self.nose = self.nose.cuda()
        self.lips = self.lips.cuda()


def upload_face_mask(path=f"{PROJECT_DIR}/FLAME/model/FLAME_masks.pkl"):
    with open(path, 'rb') as f:
        face_mask = torch.Tensor(pickle.load(f, encoding='latin1')['face']).type(torch.IntTensor)
    return face_mask


def upload_lips_mask(path=f"{PROJECT_DIR}/FLAME/model/FLAME_masks.pkl"):
    with open(path, 'rb') as f:
        lips_mask = torch.Tensor(pickle.load(f, encoding='latin1')['lips']).type(torch.IntTensor)
    return lips_mask


def upload_upper_masks(path=f"{PROJECT_DIR}/FLAME/model/FLAME_masks.pkl"):
    with open(path, 'rb') as f:
        masks_dict = pickle.load(f, encoding='latin1')
        masks = UpperMasks(forehead=masks_dict["forehead"],
                           eye_region=masks_dict["eye_region"],
                           nose=masks_dict["nose"])
    return masks


def upload_masks(path=f"{PROJECT_DIR}/FLAME/model/FLAME_masks.pkl"):
    with open(path, 'rb') as f:
        masks_dict = pickle.load(f, encoding='latin1')
        masks = Masks(forehead=torch.Tensor(masks_dict["forehead"]).type(torch.IntTensor),
                      left_eye_region=torch.Tensor(masks_dict["left_eye_region"]).type(torch.IntTensor),
                      right_eye_region=torch.Tensor(masks_dict["right_eye_region"]).type(torch.IntTensor),
                      nose=torch.Tensor(masks_dict["nose"]).type(torch.IntTensor),
                      lips=torch.Tensor(masks_dict["lips"]).type(torch.IntTensor))
    return masks
