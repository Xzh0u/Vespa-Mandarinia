import matplotlib.pyplot as plt
import os


def imgshow(train_data):
    for batch_idx, (inputs, labels) in enumerate(train_data):
        plt.figure()
        plt.imshow(inputs[batch_idx].numpy())
        plt.show()


def read_filepath(filename, root_dir):
    img_paths = []
    target = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            img_paths.append(os.path.join(root_dir, items[0]))
            target.append(int(items[1]))

    return img_paths, target


def read_path(lab_status):
    image_paths = []
    if lab_status == 'Negative ID':
        for root, dirs, files in os.walk('data/categories/Negative ID'):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))

    if lab_status == 'Positive ID':
        for root, dirs, files in os.walk('data/categories/Positive ID'):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))

    return image_paths
# train data: positive 1 negative 0 (add positive in unprocessed and unverified)
# test data: unprocessed and unverified, without label, get result manually
