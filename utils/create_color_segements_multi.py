import os
from random import shuffle
from subprocess import call, CalledProcessError
from multiprocessing.dummy import Pool


def worker(args):
    color_path, target_folder, n = args
    if os.path.isfile(os.path.join(target_folder, os.path.basename(color_path))):
        return

    try:
        call(
            ['/home/daniel/workspace/colorsegmentation/build/segmentation/bin.linux64.release/colorseg_go',
             color_path,
             '-o', target_folder,
             '-n', str(n)])
    except CalledProcessError as e:
        print(e)


def main():
    parent_folder = '/mnt/data/datasets/Imagenet/train'
    n = 50
    target_folder = '/mnt/data/datasets/Imagenet-colorsegments-n50/train'

    color_images = []
    for file in os.listdir(parent_folder):
        color_images.append((os.path.join(parent_folder, file), target_folder, n))

    shuffle(color_images)
    pool = Pool(10)
    pool.map(worker, color_images)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
