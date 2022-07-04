import random
import os
import tqdm

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from torchvision.transforms import RandomRotation, RandomAffine, \
                                   GaussianBlur, ToPILImage

basic_trans = transforms.Compose([transforms.ToTensor(),
                                  ToPILImage()])

augmentations = {'rotate': RandomRotation(15, fill=255),
                 'blur': GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                 'affine': RandomAffine(degrees=(-20, 20), translate=(0.1, 0.3), scale=(0.5, 0.75))}


# augment photo with given augmentation and saves it
def aug_save_photo(aug, aug_mane, photo_path):

    trans = transforms.Compose([transforms.ToTensor(),
                                aug,
                                ToPILImage()]
                               )

    photo = trans(Image.open(photo_path))
    new_path = photo_path[:-4] + '_' + aug_mane + '.png'
    photo.save(new_path)

    return photo


# augment photo with all augmentations and saves it
def all_aug_save_photo(photo_path):
    photo = basic_trans(Image.open(photo_path))
    photo.save(photo_path)

    aug_p = {}

    for aug_name in augmentations.keys():
        aug_p[aug_name] = aug_save_photo(augmentations[aug_name], aug_name, photo_path)

    return aug_p


# augment all photos that are original in the train/val folders
def augment_all_photos(data_path):
    classes = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    for part in ['train', 'val']:
        for name in classes:
            photo_paths = os.listdir(f'{data_path}/{part}/{name}')
            print(f'{part} photos, class = {name} augmentation')
            for path in tqdm.tqdm(photo_paths):
                augment = True
                for aug_name in augmentations.keys():
                    if aug_name in path:
                        augment = False
                        break
                if augment:
                    all_aug_save_photo(f'{data_path}/{part}/{name}/{path}')


def delete_all_aug(data_path):
    classes = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    for part in ['train', 'val']:
        for name in classes:
            photo_paths = os.listdir(f'{data_path}/{part}/{name}')
            print(f'delete {part} photos, class = {name} augmentation')
            for path in tqdm.tqdm(photo_paths):
                delete = False
                for aug_name in augmentations.keys():
                    if aug_name in path:
                        delete = True
                        break
                if delete:
                    os.remove(f'{data_path}/{part}/{name}/{path}')


# display augmentation affect with a single augmentation
def display_augmentations(photo_path):
    aug_p = all_aug_save_photo(photo_path)
    plt.title('Original')
    plt.imshow(Image.open(photo_path))
    plt.show()
    for key in aug_p.keys():
        plt.title(key)
        plt.imshow(aug_p[key])
        plt.show()
