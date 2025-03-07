# 1. Load the dataset
# 2. Use the albumentations library to augment the dataset.

import os
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip, Transpose, ElasticTransform, GridDistortion, RandomCrop, RandomBrightnessContrast


org_images_masks_dir = "/mnt/hdd_2A/train_patches/"
aug_images_masks_dir = "/mnt/hdd_2A/train_patches_augm/"


def load_data(path):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))
    return images, masks

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_data(images, masks, save_path, augment=True):
    H = 256
    W = 256

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        if augment == True:
            aug = RandomBrightnessContrast(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']
            
            aug = Transpose(p=1.0)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']
            
            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']
            
            aug = RandomCrop(H, W, p=1.0)
            augmented = aug(image=x, mask=y)
            x8 = augmented['image']
            y8 = augmented['mask']

            save_images = [x, x1, x3, x6, x7, x8]
            save_masks =  [y, y1, y3, y6, y7, y8]
            
            # save_images = [x, x4, x5, x6]
            # save_masks =  [y, y4, y5, y6]
            
            

        else:
            save_images = [x]
            save_masks = [y]

        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(images) == 1:
                tmp_img_name = f"{image_name}.{image_extn}"
                tmp_mask_name = f"{mask_name}.{mask_extn}"
            else:
                tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

            image_path = os.path.join(save_path, "images/", tmp_img_name)
            mask_path = os.path.join(save_path, "masks/", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1


""" Loading original images and masks. """
path = org_images_masks_dir
images, masks = load_data(path)
print(f"Original Images: {len(images)} - Original Masks: {len(masks)}")

# """ Creating folders. """
# create_dir("new_data/images")
# create_dir("new_data/masks")

""" Applying data augmentation. """
augment_data(images, masks, aug_images_masks_dir, augment=True)

""" Loading augmented images and masks. """
images, masks = load_data(aug_images_masks_dir)
print(f"Augmented Images: {len(images)} - Augmented Masks: {len(masks)}")