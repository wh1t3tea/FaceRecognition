import cv2
import numpy as np
from skimage import transform as trans

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def estimate_norm(lmk, image_size=112, mode='arcface'):
    """
    Estimates the normalization parameters to align facial landmarks.

    Args:
        lmk (numpy.ndarray): Array containing 5 facial landmarks.
        image_size (int): Target size of the aligned image. Default is 112.
        mode (str): Normalization mode. Default is 'arcface'.

    Returns:
        numpy.ndarray: Affine transformation matrix.
    """
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    """
    Applies the estimated normalization transformation to crop and align the input image based on the provided landmarks.

    Args:
        img (numpy.ndarray): Input image.
        landmark (numpy.ndarray): Array containing 5 facial landmarks.
        image_size (int): Target size of the aligned image. Default is 112.
        mode (str): Normalization mode. Default is 'arcface'.

    Returns:
        numpy.ndarray: Cropped and aligned image.
    """
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
