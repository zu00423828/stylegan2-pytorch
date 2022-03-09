import face_alignment
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance
import shutil
import os
from utils.face_restore_help import create_face_helper
fa = None


def fa_init():
    fa = face_alignment.FaceAlignment(
        landmarks_type=face_alignment.LandmarksType._2D, face_detector='sfd')
    return fa


def preprocess_landmark(img_list, fa, save_pkl='data.pkl'):
    rows = []
    img_list = sorted(img_list)
    for img_path in tqdm(img_list):
        img = cv2.imread(img_path)
        try:
            pred = fa.get_landmarks(img)[0]
            row = {'img_path': img_path, 'landmark': pred}
        except Exception as e:
            row = {'img_path': img_path, 'landmark': np.zeros((68, 2))}
        finally:
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_pickle(save_pkl)


def face_helper_process(img, landmark):
    face_helper.clean_all()
    face_helper.read_image(img)
    face_helper.get_face_landmarks_3(
        landmark, only_keep_largest=True, eye_dist_threshold=5)
    face_helper.align_warp_face()
    return face_helper.cropped_faces[0]


def read_data(file, out_dir):
    df = pd.read_pickle(file)
    img_path_list = df['img_path']
    landmark_list = df['landmark']
    os.makedirs(out_dir, exist_ok=True)
    for img_path, landkmark in tqdm(zip(img_path_list, landmark_list), total=len(img_path_list)):
        if 30 > distance.euclidean(landkmark[62], landkmark[66]) > 20:
            img = face_helper_process(img_path, landkmark)
            save_path = os.path.join(
                out_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, img)
            # shutil.copy(img_path, save_path)
            # print(f'{img_path} -> {save_path}')


if __name__ == '__main__':
    if fa is None:
        fa = fa_init()
    img_path = '/home/yuan/hdd/ffhq_removebg'
    save_pkl_path = 'data.pkl'
    out_dir = '/home/yuan/hdd/happy_ffhq'
    img_list = glob(f'{img_path}/*')
    if not os.path.exists(save_pkl_path):
        preprocess_landmark(img_list, fa, save_pkl_path)
    face_helper = create_face_helper(512, 1)
    read_data(save_pkl_path, out_dir)
