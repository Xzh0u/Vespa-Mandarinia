from math import *
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import folium
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score


def get_column_value(df, column_name: str):
    return np.array(df[column_name])


def main():
    dataset = pd.read_csv('dataset.csv')
    positive = dataset.loc[dataset['Lab Status'] == 'Positive ID']
    latitude = get_column_value(positive, 'Latitude').tolist()
    longitude = get_column_value(positive, 'Longitude').tolist()
    date = get_column_value(positive, 'Detection Date').tolist()

    date = pd.to_datetime(date)
    interval = (date - date[0]).days
    interval = interval - np.min(interval)

    data = []
    for i, la in enumerate(latitude):
        data.append([latitude[i], longitude[i], interval[i]])
    data = np.array(data)
    data = data[np.argsort(data[:, 2])]
    data_scale = preprocessing.scale(data)


    SSE = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data_scale)
        SSE.append(kmeans.inertia_)
    X = range(2, 9)
    plt.xlabel('Number of Clusters(k)')
    plt.ylabel('SSE')
    plt.title('SSE vs k')
    plt.plot(X, SSE, 'o-')
    plt.show()

    Scores = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        Scores.append(silhouette_score(data, kmeans.labels_, metric='euclidean'))
    X = range(2, 9)
    plt.xlabel('Number of Clusters(k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient vs k')
    plt.plot(X, Scores, 'o-')
    plt.show()

    cluster_num = 3
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(data_scale)
    label = kmeans.labels_
    centers = []
    label_list = []

    for i in range(cluster_num):
        label_list.append(data[label == i, 0:2].tolist())
        centers.append(np.mean(data[label == i], axis=0).tolist())

    centers = np.array(centers)
    centers_list = np.delete(centers, -1, axis=1).tolist()
    centers = centers[np.argsort(centers[:, 2])]
    print(centers)

    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(data[:, 1], data[:, 0], data[:, 2],
                  c=kmeans.labels_, cmap='rainbow')

    ax1.scatter3D(centers[:, 1], centers[:, 0], centers[:, 2], c='black', s=150, alpha=0.5)
    plt.show()

    x = centers[:, 1].reshape((-1, 1))
    y = centers[:, 0]

    reg = LinearRegression().fit(x, y)
    k = reg.coef_[0]
    b = reg.intercept_
    print("Y = %.5fX + (%.5f)" % (k, b))

    plt.scatter(data[:, 1], data[:, 0], c=label, cmap='rainbow')
    plt.scatter(centers[:, 1], centers[:, 0], c='black', s=150, alpha=0.5)
    data = data[np.argsort(data[:, 1])]
    plt.plot(data[np.argsort(data[:, 1])][:, 1].reshape((-1, 1)),
             reg.predict(data[np.argsort(data[:, 1])][:, 1].reshape((-1, 1))), c='b', linestyle='--')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Linear Regression of Cluster Centers(k=%d)' % cluster_num)

    plt.grid()
    plt.show()

    cluster_foot_x, cluster_foot_y = get_foot_point(centers[-1, 1], centers[-1, 0], k, b)

    print("center-%d distance to line:%.5f" % (cluster_num, get_distance([centers[-1, 1], centers[-1, 0]], [cluster_foot_x, cluster_foot_y])))
    sum_dis = 0
    for i in range(data.shape[0]):
        foot_x, foot_y = get_foot_point(data[i, 1], data[i, 0], k, b)
        sum_dis += get_distance([data[i, 1], data[i, 0]], [foot_x, foot_y])
    print("sum_dis:%.5f" % sum_dis)

    colors = ['blue', 'green', 'orange', 'pink', 'purple', 'red']
    map = folium.Map(location=[48.9938, -122.702], zoom_start=8, tiles="OpenStreetMap")
    for i in range(len(label_list)):
        point_list = label_list[i]
        for point in range(len(point_list)):
            folium.CircleMarker(radius=2.5,
                                location=label_list[i][point],
                                color=colors[i],
                                fill=True,
                                fill_color=colors[i],
                                fill_opacity=1
                                ).add_to(map)

    for i in range(len(centers_list)):
        folium.CircleMarker(cradius=6,
                            location=centers_list[i],
                            color=colors[i],
                            fill=True,
                            fill_color=colors[i],
                            fill_opacity=0.3
                            ).add_to(map)

    map.save('map_cluster%d.html' % cluster_num)


def get_foot_point(point_x, point_y, k, b):
    foot_x = (point_x + k * (point_y - b)) / (k * k + 1)
    foot_y = k * foot_x + b
    return foot_x, foot_y


def get_distance(origin, destination):
    lon1 = radians(float(destination[0]))
    lon2 = radians(float(origin[0]))
    lat1 = radians(float(destination[1]))
    lat2 = radians(float(origin[1]))
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dist = 2 * asin(sqrt(a)) * 6371 * 1000
    return dist

if __name__ == "__main__":
    main()
