import numpy as np
import cv2 as cv
import csv
import os
np.set_printoptions(threshold=np.inf)
def read_datasets(img_path, density_map_path, scale=4):
    head_count = img_path.split(r'/')[-1].split(r'.')[0].split(r'_')[-1]
    head_count = np.asarray(int(head_count), dtype=np.float32)
    im = cv.imread(img_path) * 1.0 / 255.0
    im = np.asarray(im, dtype=np.float32)
    density_map = np.loadtxt(open(density_map_path, 'rb'), dtype=np.float32, delimiter=",", skiprows=0)
    density_map = cv.resize(density_map, (0, 0), fx=1.0 / scale, fy=1.0 / scale, interpolation=cv.INTER_CUBIC)

    im = im.reshape((1, im.shape[0], im.shape[1], 3))
    density_map = density_map.reshape((1, density_map.shape[0], density_map.shape[1], 1))

    head_count = head_count.reshape((1,1))
    return [im, density_map, head_count]

if __name__ == '__main__':
    img_root_dir = './datasets/formatted_trainval/shanghaitech_part_A_patches_9/train/'
    den_root_dir = './datasets/formatted_trainval/shanghaitech_part_A_patches_9/train_den/'

    # 列出文件夹下所有的目录与文件
    list = os.listdir(img_root_dir)
    img_path = img_root_dir + list[0]
    density_map_path = den_root_dir + str(list[0]).split(r'.')[0] + r'.csv'

    [img, density_map, head_count] = read_datasets(img_path, density_map_path)
    print(img.shape)
    print(density_map.shape)
