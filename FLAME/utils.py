import torch
import pickle


def upload_face_mask(path="C:/Content/Python/AvatarAnimation/FLAME/model/FLAME_masks.pkl"):
    with open(path, 'rb') as f:
        face_mask = torch.Tensor(pickle.load(f, encoding='latin1')['face']).type(torch.IntTensor)
    return face_mask


def upload_lips_mask(path="C:/Content/Python/AvatarAnimation/FLAME/model/FLAME_masks.pkl"):
    with open(path, 'rb') as f:
        lips_mask = torch.Tensor(pickle.load(f, encoding='latin1')['lips']).type(torch.IntTensor)
    return lips_mask
