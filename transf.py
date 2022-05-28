from params import RANDOM_SEED, LocationConfig, CreateDataConfig
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pathlib import Path
import face_recognition
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import insightface
import fnmatch
import pickle
import torch
import cv2
import os

def create_new_data_directories(path):
    Path(path).mkdir(exist_ok=True, parents=True)
    Path(path + 'train').mkdir(exist_ok=True, parents=True)
    Path(path + 'test').mkdir(exist_ok=True, parents=True)
    Path(LocationConfig.crop_images).mkdir(exist_ok=True, parents=True)
    
def get_short_video_name(videoNames):
    ShortVideoName = []
    for videoName in videoNames.values:
        ShortVideoName.append(videoName.split('.')[0])
    return ShortVideoName

def create_mean_video_name_df(df):
    cols = ['ValueExtraversion','ValueAgreeableness','ValueConscientiousness','ValueNeurotisicm','ValueOpenness','ShortVideoName']
    grouped_df = df[cols].groupby('ShortVideoName')
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()
    return mean_df


def create_dataset_ChaLearn(images_path, end_path):
    create_new_data_directories(end_path)
    df = pd.read_csv(LocationConfig.labels + 'bigfive_labels.csv')

    df['ShortVideoName'] = get_short_video_name(df['VideoName'])

    mean_df = create_mean_video_name_df(df)
    mean_df.to_csv(LocationConfig.labels + 'bigfive_labels_mean.csv')
    df = mean_df.set_index('ShortVideoName')
    all_x = np.array(df.index)
    X_train, X_test = train_test_split(
        all_x, 
        test_size=CreateDataConfig.test_size_ratio,
        random_state=0
    )

    images_dict_train = {'X':[], 'Y':[]}
    images_dict_test = {'X':[], 'Y':[]}
    total_files = len(fnmatch.filter(os.listdir(images_path), '*.jpg'))

    for image_path in tqdm(Path(images_path).glob('*.jpg'), total=total_files):
        img = face_recognition.load_image_file(str(image_path))
        face_locations = face_recognition.face_locations(img, model="cnn")
        codes = face_recognition.face_encodings(img, known_face_locations=face_locations, model="large")
        if len(codes) < 1:
            continue
        image_group = image_path.name.split('.')[0]
        Y = df.loc[image_group].values
        image_no = image_path.name.split('.')[2][-5:]
        if CreateDataConfig.classification:
            Y = list(np.where(Y>CreateDataConfig.Y_threshold, 1, 0))
        if image_group in X_test:
            images_dict_test['X'].append(codes[0])
#             images_dict_test['X'].append(img)
            images_dict_test['Y'].append(Y)
        else:
            images_dict_train['X'].append(codes[0])
#             images_dict_train['X'].append(img)
            images_dict_train['Y'].append(Y)
            
    with open(end_path + 'train/train.pickle', 'wb') as handle:
        pickle.dump(images_dict_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(end_path + 'test/test.pickle', 'wb') as handle:
        pickle.dump(images_dict_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
create_dataset_ChaLearn(LocationConfig.crop_images, LocationConfig.enc)