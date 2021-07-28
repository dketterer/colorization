import os
import json

from colorization.data import is_rgb


def main():
    base_path = '/media/daniel/Windowsdata/Imagenet/test'
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']

    imagenet_paths = []
    file_names = []

    for file in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, file)):
            for file2 in os.listdir(os.path.join(base_path, file)):
                if os.path.splitext(file2)[1] in image_extensions:
                    if is_rgb(os.path.join(base_path, file, file2)):
                        imagenet_paths.append(os.path.join(base_path, file, file2))
                        file_names.append(file2)
        if os.path.splitext(file)[1] in image_extensions:
            if is_rgb(os.path.join(base_path, file)):
                imagenet_paths.append(os.path.join(base_path, file))
                file_names.append(file)

    with open('/media/daniel/Windowsdata/Imagenet/test_rgb_images.json', 'w+') as f:
        json.dump(file_names, f)


if __name__ == '__main__':
    main()
