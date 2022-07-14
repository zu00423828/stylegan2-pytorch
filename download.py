import cv2
import numpy as np
import requests
import os
import time
from tqdm import trange, tqdm
import multiprocessing


def download_avatar(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in trange(10000):
        save_name = f'{save_dir}/{i+1:06d}.png'
        if os.path.exists(save_name):
            continue
        url = "https://thispersondoesnotexist.com/image"
        r = requests.get(
            url, headers={'User-Agent': "My User Agent 1.0"}).content
        image = np.frombuffer(r, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (1024, 1024))
        cv2.imwrite(save_name, image)
        time.sleep(0.05)


def download_avatar_only(save_name):
    i, name = save_name
    if os.path.exists(name):
        return
    url = "https://thispersondoesnotexist.com/image"
    r = requests.get(
        url, headers={'User-Agent': "My User Agent 1.0"}).content
    image = np.frombuffer(r, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (1024, 1024))
    cv2.imwrite(name, image)
    time.sleep(0.05)
    # return image, save_name
    return i, name


def multi_download(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_names = [(i, f'{save_dir}/{i+1:06d}.png') for i in range(10000)]
    with multiprocessing.Pool(4) as pool:
        for i, name in tqdm(pool.imap_unordered(download_avatar_only, save_names)):
            # cv2.imwrite(save_name, image)
            print(i, name)


if __name__ == "__main__":
    save_dir = '/home/yuan/hdd8t/high_face/src4'
    download_avatar(save_dir)
    # save_dir = '/home/yuan/hdd8t/high_image_test'
    # multi_download(save_dir)
