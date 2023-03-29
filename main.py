from animation_by_video.detector.detector import FaceTracker
import torch
import numpy as np
from FLAME.flame_model import RADIAN
from animation_by_video.visualizer.visualizer import Visualizer
from FLAME.flame_model import FlameModel
from FLAME.config import get_config


# tracker = FaceTracker()
# tracker.execute()
# vis = Visualizer()
# fm = FlameModel(get_config())
# shape_params = torch.zeros(1, 100).cuda()  # 35, 45, 90, 0..30, -4..5, -6..8
# pose_params_numpy = np.array([[0, 0, 0, 25 * RADIAN, 0 * RADIAN, 0 * RADIAN]], dtype=np.float32)
# pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()
# expression_params = torch.zeros(1, 50, dtype=torch.float32).cuda()
# v, l = fm.generate(shape_params, pose_params, expression_params)
# vis.set_surfaces(fm.flamelayer.faces)
# vis.render(v.detach().cpu().numpy().squeeze())
