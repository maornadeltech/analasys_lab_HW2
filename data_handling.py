import torch
import os
import random
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from itertools import permutations
from torchvision import transforms
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps

import data_augmenting


TRAIN_PATH = '/home/student/Desktop/HW2/HW2_code/data/train'
PHOTO_SIZE = 256
all_locations = list(permutations(range(PHOTO_SIZE), 2)) + list([(i, i) for i in range(PHOTO_SIZE)])
trans = transforms.Compose([transforms.ToTensor(),
                           transforms.Resize((PHOTO_SIZE, PHOTO_SIZE))]
                           )
random.seed(10)


def transfer_photos(transfer_path, origin_path, transfer_size=0.2):
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']

    if not os.path.exists(transfer_path):
        os.mkdir(transfer_path)

    for name in classes:
        if not os.path.exists(f'{transfer_path}/{name}'):
            os.mkdir(f'{transfer_path}/{name}')

    for name in classes:
        images = os.listdir(f'{origin_path}/{name}')

        if transfer_size > 1:
            photo_idx = random.sample(list(range(len(images))), transfer_size)
        else:
            photo_idx = random.sample(list(range(len(images))), int(len(images)*transfer_size))
        for i in photo_idx:
            image = images[i]
            shutil.move(f'{origin_path}/{name}/{image}', f'{transfer_path}/{name}/{image}')


def create_plot(paths, points, title, zoom=0.05, fig_size=[10.0, 10.0]):
    plt.rcParams["figure.figsize"] = fig_size
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1])
    for i in range(len(paths)):
        photo = OffsetImage(ImageOps.grayscale(Image.open(paths[i]).convert('L')), zoom=zoom)
        ab = AnnotationBbox(photo, (points[i][0], points[i][1]), frameon=False)
        ax.add_artist(ab)
    plt.title(title)
    plt.show()


def plot_cluster_points(pca_dim=2):
    classes = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    all_photos = []
    all_paths = []
    lengths = [0]
    colors = list(mcolors.TABLEAU_COLORS.keys())
    curr_sum = 0
    for name in classes:
        photos = []
        p_path = []
        data_dir = f'{TRAIN_PATH}/{name}'

        for file in os.listdir(data_dir):
            p = Image.open(f'{data_dir}/{file}')
            p_tensor = trans(p).flatten()
            photos.append(p_tensor.detach().cpu().numpy())
            p_path.append(f'{data_dir}/{file}')

        photos = np.array(photos)
        pca = PCA(n_components=pca_dim)
        pca.fit(photos)
        points = pca.transform(photos)
        t = f'Distribution for pca of dim 2 for {name}:'
        create_plot(p_path, points, t)

        all_photos.extend(photos)
        all_paths.extend(p_path)
        curr_sum += len(photos)
        lengths.append(curr_sum)

    pca = PCA(n_components=pca_dim)
    pca.fit(all_photos)

    for i in range(1, 11):
        points = pca.transform(all_photos[lengths[i-1]: lengths[i]])
        plt.scatter(points[:, 0], points[:, 1], c=colors[i-1], label=classes[i-1])

    plt.title('Plot of pca for all classes')
    plt.legend()
    plt.show()


def run_DBSCAN(photo_t):
    locations = torch.tensor(all_locations)
    values = photo_t.flatten()
    vecs = torch.concat([locations, values.unsqueeze(dim=1)], dim=1)
    model = DBSCAN(eps=1, min_samples=2)
    model.fit(vecs)
    labels = model.labels_
    res_img = torch.tensor(labels.reshape(PHOTO_SIZE, PHOTO_SIZE))
    plt.imshow(res_img)
    plt.show()


def run_all_dbscan():
    classes = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    for name in classes:
        data_dir = f'{TRAIN_PATH}_old/{name}'

        for file in os.listdir(data_dir):
            p = Image.open(f'{data_dir}/{file}')
            plt.imshow(p)
            plt.show()
            p_tensor = trans(p)
            run_DBSCAN(p_tensor)
            break


if __name__ == '__main__':
    data_augmenting.delete_all_aug('/home/student/Desktop/HW2/HW2_code/data')
    transfer_photos(TRAIN_PATH, '/home/student/Desktop/HW2/HW2_code/data/val', 25)
    transfer_photos('/home/student/Desktop/HW2/HW2_code/data/val', TRAIN_PATH)
    data_augmenting.augment_all_photos('/home/student/Desktop/HW2/HW2_code/data')

