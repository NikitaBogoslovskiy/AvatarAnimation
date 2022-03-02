from imutils import face_utils
import dlib
import cv2
import os


class Detector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./files/shape_predictor.dat')

    def get_image(self, image):
        # self.cropped_image = imutils.resize(image, width=500)
        self.cropped_image = image
        self.gray_image = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)
        self.corrected_image = self.gray_image

    def load_image(self, path):
        if not os.path.exists(path):
            raise IOError('Path is incorrect')
        self.original_image = cv2.imread(path)
        self.get_image(self.original_image)

    def detect_face(self):
        rects = self.detector(self.corrected_image, 1)
        if len(rects) == 0:
            return False
        else:
            self.rect = rects[0]
            return True

    def detect_landmarks(self):
        shape = self.predictor(self.corrected_image, self.rect)
        self.landmarks = face_utils.shape_to_np(shape)[17:]

    def visualize_box(self):
        self.new_image = self.cropped_image.copy()
        x, y, w, h = face_utils.rect_to_bb(self.rect)
        cv2.rectangle(self.new_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    def visualize_landmarks(self):
        for xy in self.landmarks:
            cv2.circle(self.new_image, xy, 1, (0, 0, 255), -1)

    def show(self):
        cv2.imshow('Landmarks Detector: the result', self.new_image)
        cv2.waitKey(0)

    def save(self, path):
        cv2.imwrite(path, self.new_image)


def track_face(path=0):
    det = Detector()
    video = cv2.VideoCapture(path)
    while True:
        ret, frame = video.read()
        det.get_image(frame)
        if det.detect_face():
            det.visualize_box()
            det.detect_landmarks()
            det.visualize_landmarks()
            frame = det.new_image
        cv2.imshow('Video with box and landmarks', frame)
        if cv2.waitKey(1) != -1:
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
