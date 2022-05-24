import os

import insightface
import numpy as np

model_detector_path = './det_10g.onnx'


class FaceDetection:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.image_size = 640
        self.det_thresh = 0.7
        self.detector = self._get_face_detection_model()

    def _get_device_used_code(self):
        if not self.use_gpu:
            return -1
        return 1

    def _get_face_detection_model(self):
        app_kwargs = {'providers': ['CUDAExecutionProvider']}
        detector = insightface.model_zoo.get_model(model_detector_path, **app_kwargs)
        detector.prepare(ctx_id=self._get_device_used_code(), det_thresh=self.det_thresh, input_size=(self.image_size, self.image_size))
        return detector

    def detect_face(self, img: np.ndarray) -> np.ndarray:
        """
        Detects a face in an image

        Parameters
        ----------
        img
            Input image with a face
        Returns
        -------
        Bounding box of the original face
        """
        bboxes, _ = self.detector.detect(img)
        return bboxes


face_detector = FaceDetection()
