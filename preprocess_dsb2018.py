from skimage.io import imread, imsave
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings

def main():
    img_size = 96

    paths = glob('input/dsb2018/stage1_train/*')

    if not os.path.exists('input/dsb2018_%d/images'%img_size):
        os.makedirs('input/dsb2018_%d/images'%img_size)
    if not os.path.exists('input/dsb2018_%d/masks'%img_size):
        os.makedirs('input/dsb2018_%d/masks'%img_size)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for i in tqdm(range(len(paths))):
            path = paths[i]
            image = imread(path+'/images/'+os.path.basename(path)+'.png')
            mask = np.zeros((image.shape[0], image.shape[1]))
            for mask_path in glob(path+'/masks/*'):
                mask_ = imread(mask_path) > 127
                #mask = mask | mask_
                #mask = np.maximum(mask, mask_)
                mask[mask_] = 1
            if (image.shape[2] == 4):
                image = image[:,:,:3]
            if (len(image.shape) == 2):
                image = np.tile(image[:,:,np.newaxis], (1, 1, 3))
            image = cv2.resize(image, (img_size, img_size))
            mask = cv2.resize(mask, (img_size, img_size))
            imsave('input/dsb2018_%d/images/'%img_size+os.path.basename(path)+'.png', image)
            imsave('input/dsb2018_%d/masks/'%img_size+os.path.basename(path)+'.png', (mask*255).astype('uint8'))


if __name__ == '__main__':
    main()
