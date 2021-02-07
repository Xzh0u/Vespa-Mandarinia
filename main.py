import pandas as pd
import numpy as np
import os
from shutil import copyfile
# import folium


def get_column_value(df, column_name: str):
    return np.array(df[column_name])


def main():
    dataset = pd.read_csv('data/dataset.csv')
    # images = pd.read_csv('data/images_by_globalID.csv')
    unverified = dataset.loc[dataset['Lab Status'] == 'Unverified']
    unprocessed = dataset.loc[dataset['Lab Status'] == 'Unprocessed']
    negative = dataset.loc[dataset['Lab Status'] == 'Negative ID']
    positive = dataset.loc[dataset['Lab Status'] == 'Positive ID']
    print("\n--- Lab Status Count ---")
    print("Positive ID: ", positive.shape[0])
    print("Negative ID: ", negative.shape[0])
    print("Unverified:  ", unverified.shape[0])
    print("Unprocessed: ", unprocessed.shape[0])
    # image_gID = get_column_value(images, 'GlobalID').tolist()
    # filename = get_column_value(images, 'FileName').tolist()
    # filetype = get_column_value(images, 'FileType').tolist()
    # dataset_gID = get_column_value(unprocessed, 'GlobalID').tolist(
    # ) + get_column_value(unverified, 'GlobalID').tolist()
    # print(len(dataset_gID))

    # img_name_list = []

    # for gid in dataset_gID:
    #     for i, iid in enumerate(image_gID):
    #         if iid == gid:
    #             if filetype[i] == 'image/jpg' or filetype[i] == 'image/png':
    #                 img_name_list.append(filename[i])

    # print(img_name_list)

    # for file in os.listdir("data/imgs/"):
    #     for i in range(0, len(img_name_list)):
    #         if file == img_name_list[i]:
    #             copyfile('imgs/' + file, 'imgs/tmp/' + file)


if __name__ == "__main__":
    main()
