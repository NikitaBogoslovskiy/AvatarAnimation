from video_animation.visualizer.offline_visualizer import OfflineVisualizer


class VisualizerParams:
    def __init__(self,
                 save_path,
                 surfaces,
                 width,
                 height,
                 frame_rate):
        self.save_path = save_path
        self.surfaces = surfaces
        self.width = width
        self.height = height
        self.frame_rate = frame_rate


def render_sequentially(params: VisualizerParams, queue):
    v = OfflineVisualizer(params.save_path)
    v.set_surfaces(params.surfaces)
    v.init_settings(animation_resolution=(params.height, params.height),
                    input_resolution=(params.width, params.height),
                    frame_rate=params.frame_rate)
    while True:
        if queue.empty():
            continue
        top = queue.get()
        if top == -1:
            break
        v.render(top[0], top[1])
    v.release()
