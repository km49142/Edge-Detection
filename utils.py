### utils.py
import cv2
import numpy as np
import os
import scipy.io


def load_image_and_mask(image_path, mask_path, target_size=(128, 128)):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = img[..., np.newaxis]

    # Load .mat edge mask (BSDS format)
    mat = scipy.io.loadmat(mask_path)
    # Each 'groundTruth' entry is a dict with 'Boundaries' and 'Segmentation'
    mask = mat['groundTruth'][0][0][0][0][1]  # Get 'Boundaries' field from the first annotation

    # Resize and normalize
    mask = cv2.resize(mask.astype(np.float32), target_size)
    mask = mask[..., np.newaxis]

    print(type(mat['groundTruth'][0][0][0]))   # Should be a numpy.void
    print(mat['groundTruth'][0][0][0][0].dtype.names)  # Should show ('Segmentation', 'Boundaries')


    return img, mask


    return img, mask

def preprocess_image(path, target_size=(128, 128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = img[..., np.newaxis]
    return img
