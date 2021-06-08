import os
import cv2

if __name__ == '__main__':
    img_folder = '/mnt/data/datasets/color-overfit/n01440764'
    for i, img_path in enumerate(os.listdir(img_folder)):
        img = cv2.imread(os.path.join(img_folder, img_path))
        img_small = cv2.resize(img, (256, 256))
        cv2.imwrite(os.path.join(img_folder, f'example-{i}.jpg'), img_small)
