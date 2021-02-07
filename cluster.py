from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_column_value(df, column_name: str):
    return np.array(df[column_name])


def main():
    dataset = pd.read_csv('dataset.csv')
    latitude = get_column_value(dataset, 'Latitude').tolist()
    longitude = get_column_value(dataset, 'Longitude').tolist()
    position = []
    for i, la in enumerate(latitude):
        position.append([latitude[i], longitude[i]])

    kmeans = KMeans(n_clusters=2, random_state=0).fit(position)
    print(kmeans.cluster_centers_)

    plt.style.use("fivethirtyeight")
    plt.scatter(position[:][0], position[:][1],
                c=kmeans.labels_, cmap='rainbow')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()


main()
