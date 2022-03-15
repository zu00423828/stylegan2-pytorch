import face_alignment
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance
import shutil
import os
import click
from utils.face_restore_help import create_face_helper
from utils.utils import EasyDict


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


def face_helper_process(img, landmark, face_helper):
    face_helper.clean_all()
    face_helper.read_image(img)
    face_helper.get_face_landmarks_3(
        landmark, only_keep_largest=True, eye_dist_threshold=5)
    face_helper.align_warp_face()
    return face_helper.cropped_faces[0]


def read_data(file, out_dir, face_process=True):
    face_helper = None
    df = pd.read_pickle(file)
    img_path_list = df['img_path']
    landmark_list = df['landmark']
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for img_path, landkmark in tqdm(zip(img_path_list, landmark_list), total=len(img_path_list)):
        if 30 >= distance.euclidean(landkmark[62], landkmark[66]) >= 20:
            save_path = os.path.join(
                out_dir, os.path.basename(img_path))
            if face_process:
                if face_helper is None:
                    face_helper = create_face_helper(512, 1)
                img = face_helper_process(img_path, landkmark, face_helper)
                cv2.imwrite(save_path, img)
            else:
                shutil.copy(img_path, save_path)
                # print(f'{img_path} -> {save_path}')
            count += 1
    print(count)


@click.command()
@click.option('--image_path', type=str, required=True)
@click.option('--save_pkl_path', type=str, required=True)
@click.option('--out_dir', type=str, required=True)
@click.option('--face_process', is_flag=True, default=False, show_default=True)
def main(**kwargs):
    args = EasyDict(kwargs)
    img_path = args.image_path
    save_pkl_path = args.save_pkl_path
    out_dir = args.out_dir
    face_process = args.face_process
    img_list = glob(f'{img_path}/*')
    fa = None
    if not os.path.exists(save_pkl_path):
        if fa is None:
            fa = fa_init()
        preprocess_landmark(img_list, fa, save_pkl_path)
    read_data(save_pkl_path, out_dir, face_process=face_process)


if __name__ == '__main__':
    main()
