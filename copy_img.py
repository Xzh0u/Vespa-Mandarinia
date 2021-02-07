import pandas as pd
import numpy as np
from shutil import copyfile


def get_column_value(df, column_name: str):
    return np.array(df[column_name])


def add_image_mapping(mappings, gid, filename, status):
    if gid not in mappings:
        mappings[gid] = []

    mappings[gid].append({'filename': filename, 'status': status})


def category_mapping(mapping):
    for item in mapping:
        file_path = f'./data/imgs/{item["filename"]}'
        target_path = f'./data/categories/{item["status"]}/{item["filename"]}'
        copyfile(file_path, target_path)


def main():
    dataset = pd.read_csv('data/dataset.csv')
    images = pd.read_csv('data/images_by_globalID.csv')

    lab_statuses = get_column_value(dataset, 'Lab Status').tolist()
    image_globalIDs = get_column_value(images, 'GlobalID').tolist()
    filenames = get_column_value(images, 'FileName').tolist()

    dataset_gID = get_column_value(dataset, 'GlobalID').tolist()

    print(len(dataset_gID))

    image_mappings = {}

    for gID_idx, gid in enumerate(dataset_gID):
        for i, iid in enumerate(image_globalIDs):
            if iid == gid:
                add_image_mapping(image_mappings, gid,
                                  filenames[i], lab_statuses[gID_idx])

    print(len(image_mappings))

    for gid in image_mappings:
        mapping = image_mappings[gid]
        category_mapping(mapping)

    print('success')


main()
