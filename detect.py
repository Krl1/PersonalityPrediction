# python detect.py -i <path_to_folder_with_images> -o <path_to_folder_for_crop_images>

import argparse
from pathlib import Path
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import fnmatch
from face_detection import FaceDetection
from params import RANDOM_SEED, LocationConfig, CreateDataConfig


def load_img(path: str) -> np.ndarray:
    if not Path(path).exists():
        raise Exception(f"Given file: '{path}' - does not exists!")
    return cv2.imread(path)


def squarify(img):
    old_image_height, old_image_width, channels = img.shape
    if min(old_image_height, old_image_width) < 10:
        return None
    max_size = max(old_image_height, old_image_width)
    if max_size < 38:
        return None
    color = (255,255,255)
    result = np.full((max_size,max_size, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (max_size - old_image_width) // 2
    y_center = (max_size - old_image_height) // 2
    
    x_error = (max_size - old_image_width) % 2
    y_error = (max_size - old_image_height) % 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height+x_error, 
           x_center:x_center+old_image_width+y_error] = img
    
    return result


def detect_and_crop(face_detect_app: FaceDetection, input_path: str, output_path: str):
    img = load_img(input_path)
    faces = face_detect_app.detect_face(img)
    if len(faces) == 0 or len(faces) > 1:
        return

    face_box = faces[0].astype(np.int32)
    try:
        new_img = img[face_box[1] : face_box[3], face_box[0] : face_box[2]]
    except IndexError:
        return

    new_img = squarify(new_img)
    if new_img is None:
        return
    
    filename = input_path.split('/')[-1]
    cv2.imwrite(os.path.join(output_path,filename), new_img)


if __name__ == "__main__":    
    face_detector = FaceDetection()
    
    total_f=0
    for image_path in tqdm(Path(LocationConfig.raw_data).glob('*/*.jpg')):
        total_f += 1
        
    for image_path in tqdm(Path(LocationConfig.raw_data).glob('*/*.jpg'), total=total_f):
        detect_and_crop(face_detector, image_path, LocationConfig.crop_data)
