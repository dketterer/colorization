import timeit

import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import label2rgb
from cuda_slic import slic

region_size = 40
ruler = 10
img_path = 'resources/ILSVRC2012_val_00000014.JPEG'
img = cv2.imread(img_path)
img = cv2.GaussianBlur(img, (7, 7), 1)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def create_slico():
    slico: cv2.ximgproc_SuperpixelSLIC = cv2.ximgproc.createSuperpixelSLIC(img_lab, algorithm=cv2.ximgproc.SLICO,
                                                                           region_size=region_size, ruler=ruler, )
    slico.iterate(5)
    slico.enforceLabelConnectivity()

    labels = slico.getLabels()
    return labels


def cuda_slic():
    labels = slic(img_lab, n_segments=100, compactness=5, max_iter=5, convert2lab=False, enforce_connectivity=False)
    return labels


def main():
    labels = cuda_slic()
    # labels = create_slico()
    npix = np.max(labels)

    print(npix)
    from matplotlib import cm
    c_dict = cm.get_cmap('jet')._segmentdata
    jet_cm = LinearSegmentedColormap('foo', segmentdata=c_dict, N=npix)
    colors = cm.get_cmap(jet_cm, npix)(np.arange(npix))
    np.random.shuffle(colors)

    L = label2rgb(labels, colors=colors, bg_label=-1)

    cv2.imshow('slico', L)
    cv2.imshow('original', img)
    cv2.waitKey()


if __name__ == '__main__':
    # main()
    # print(timeit.timeit('create_slico()', globals=locals(), number=50) / 50)
    print(timeit.timeit('cuda_slic()', globals=locals(), number=1000) / 1000)
    # both with 5 iters:
    # 0.03544591887970455
    # 0.03716413646005094
