from imutils import face_utils
import dlib
import cv2
import os


class Detector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmarks_detector = dlib.shape_predictor('animation_by_video/detector/shape_predictor.dat')
        self.image = None
        self.draft_image = None
        self.bounding_box = None
        self.landmarks = None

    def get_image(self, image):
        self.image = image
        self.draft_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def load_image(self, path):
        if not os.path.exists(path):
            raise Exception('Image path is incorrect')
        self.get_image(cv2.imread(path))

    def detect_face(self):
        bounding_boxes = self.face_detector(self.draft_image, 1)
        if len(bounding_boxes) == 0:
            return False, None
        else:
            self.bounding_box = bounding_boxes[0]
            return True, self.bounding_box

    def detect_landmarks(self):
        shape = self.landmarks_detector(self.draft_image, self.bounding_box)
        self.landmarks = face_utils.shape_to_np(shape)
        return self.landmarks

    def visualize_bounding_box(self):
        x, y, w, h = face_utils.rect_to_bb(self.bounding_box)
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    def visualize_landmarks(self):
        for x, y in self.landmarks[0:]:
            cv2.circle(self.image, (x, y), 2, (0, 0, 255), -1)

    def show(self):
        cv2.imshow('Landmarks Detector: the result', self.image)
        cv2.waitKey(0)

    def save(self, processed_image_path):
        cv2.imwrite(processed_image_path, self.image)


class FaceTracker:
    def __init__(self):
        self.detector = Detector()

    def execute(self, video_path=0):
        video = cv2.VideoCapture(video_path)
        while True:
            _, frame = video.read()
            self.detector.get_image(frame)
            face_detected, _ = self.detector.detect_face()
            if face_detected:
                self.detector.visualize_bounding_box()
                self.detector.detect_landmarks()
                self.detector.visualize_landmarks()
            cv2.imshow('Detection', self.detector.image)
            if cv2.waitKey(1) != -1:
                break
        video.release()
        cv2.destroyAllWindows()
