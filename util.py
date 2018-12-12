import cv2 
import numpy as np
import random


def generator(X, y, size):
    '''
    generates randomly sampled data from given set
    X, y - source data, source labels
    size - number of data samples
    '''
    # store data in dictionary
    X_dict = {i:[] for i in np.unique(y)}
    for i in range(len(X)):
        X_dict[y[i]].append(X[i])

    # generator
    while True:
        # sample y
        y_out = np.random.choice(np.unique(y), size=size)
        # sample x
        x_out = []
        for i, cat in enumerate(y_out):
            x = random.choice(X_dict[cat])
            x_out.append(x)
        x_out = np.stack(x_out)

        yield x_out, y_out

# preprocessing function
def grayscale_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def normalize_img(img):
    '''
    rescale pixel values to -0.5 to 0.5 range
    '''
    # shift center and normalize
    _img = img.copy()
    norm = (_img.astype(np.float32) - 127)/255.0
    return norm

def normalize(X, gray=False):
    n = len(X)
    if gray:
        X_out = np.zeros(list(X.shape[:-1]) + [1]).astype(np.float32)
    else:
        X_out = np.zeros_like(X).astype(np.float32)
    for i in range(n):
        if gray:
            X_out[i] = normalize_img(X[i])[:,:, np.newaxis]
        else:
            X_out[i] = normalize_img(X[i])
    return X_out

def random_transforms(X, 
               gray=False, 
               distort=False, 
               resize=False, 
               rotate=False, 
               shift=False, 
               blur_or_sharp=False):

    # prepare pipeline
    pipeline = []
    if gray: 
        pipeline.append(grayscale_img)
    if distort: 
        pipeline.append(distort_img)
    if resize: 
        pipeline.append(resize_img)
    if rotate:
        pipeline.append(rotate_img)
    if shift:
        pipeline.append(shift_img)

    # prepare output array
    n = len(X)
    if gray:
        X_out = np.zeros(list(X.shape[:-1]) + [1], dtype=np.uint8)
    else:
        X_out = np.zeros_like(X, dtype=np.uint8)

    # execute processing
    for i in range(n):
        if gray:
            X_out[i] = run_pipeline(pipeline, X[i])[:,:, np.newaxis]
        else:
            X_out[i] = run_pipeline(pipeline, X[i])
    return X_out

def run_pipeline(pipeline, img):
    x = img.copy()
    for f in pipeline:
        try:
            y = f(x)
            x = y
        except:
            print('error with ', f.__name__)
    return y

# image augmentation functions
# taken from: 
# https://github.com/dingran/traffic-sign-classifier/blob/master/tsc_utils.py
def distort_img(input_img, d_limit=4):
    """
    Apply warpPerspective transformation on image, with 4 key points, randomly generated around the corners
    with uniform distribution with a range of [-d_limit, d_limit]
    :param input_img:
    :param d_limit:
    :return:
    """
    if d_limit == 0:
        return input_img
    rows, cols, ch = input_img.shape
    pts2 = np.float32([[0, 0], [rows - 1, 0], [0, cols - 1], [rows - 1, cols - 1]])
    pts1 = np.float32(pts2 + np.random.uniform(-d_limit, d_limit, pts2.shape))
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(input_img, M, (cols, rows), borderMode=1)
    return dst


def resize_img(input_img, scale=1.1):
    """
    Function to scale image content while keeping the overall image size, padding is done with border replication
    Scale > 1 means making content bigger
    :param input_img: X * Y * ch
    :param scale: positive real number
    :return: scaled image
    """
    if scale == 1.0:
        return input_img
    rows, cols, ch = input_img.shape
    d = rows * (scale - 1)  # overall image size change from rows, cols, to rows - 2d, cols - 2d
    pts1 = np.float32([[d, d], [rows - 1 - d, d], [d, cols - 1 - d], [rows - 1 - d, cols - 1 - d]])
    pts2 = np.float32([[0, 0], [rows - 1, 0], [0, cols - 1], [rows - 1, cols - 1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(input_img, M, (cols, rows), borderMode=1)
    return dst


def rotate_img(input_img, angle=15):
    if angle == 0:
        return input_img
    rows, cols, ch = input_img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(input_img, M, (cols, rows), borderMode=1)
    return dst


def shift_img(input_img, dx=2, dy=2):
    if dx == 0 and dy == 0:
        return input_img
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(input_img, M, (input_img.shape[0], input_img.shape[1]), borderMode=1)
    return dst


def blur_or_sharpen_img(input_img, kernel=(3, 3), ratio=0.7, factor=1.0):
    blur = cv2.GaussianBlur(input_img, kernel, 0)
    sharp = cv2.addWeighted(input_img, 1.0 + ratio * factor, blur, -ratio * factor, 0)
    return np.random.choice([blur, sharp])

