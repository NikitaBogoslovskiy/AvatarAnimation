from face_processing import landmarks
import training.dataset as ds
import training.training


landmarks.track_face()
# ds.generate('files/mini_dataset.pkl', 5)
# training.training.train_model()
