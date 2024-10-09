import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pacmap
import os
import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.manifold import TSNE
import trimap
import umap

os.environ['OMP_NUM_THREADS'] = '1'

def visualize_with_pacmap(scaled_data, title):
    n_components = 2
    n_neighbors = 3
    MN_ratio = 2.0
    FP_ratio = 3.0
    embedding = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio)

    pacmac_result = embedding.fit_transform(scaled_data, init="pca")

    plt.figure(figsize=(8, 6))
    plt.scatter(pacmac_result[:, 0], pacmac_result[:, 1], c='blue', cmap='Spectral')
    plt.title(title)
    plt.xlabel("age")
    plt.ylabel("serum cholestoral")
    # Добавляем текст с информацией о переменных
    info_text = f'n_components: {n_components}\nn_neighbors: {n_neighbors}\nMN_ratio: {MN_ratio}\nFP_ratio: {FP_ratio}'
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.show()

def visualize_with_tsne(scaled_data, title):
    n_components = 2
    perplexity = 20
    learning_rate = 400
    max_iter = 100
    random_state = 42
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter, random_state=random_state)
    tsne_result = tsne.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='blue', cmap='Spectral')
    plt.title(title)
    plt.xlabel("age")
    plt.ylabel("serum cholestoral")
    # Добавляем текст с информацией о переменных
    info_text = f'n_components: {n_components}\nperplexity: {perplexity}\nlearning_rate: {learning_rate}\nmax_iter: {max_iter}\nrandom_state: {random_state}'
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

def visualize_with_trimap(scaled_data, title):
    n_inliers = 22
    n_outliers = 40
    n_random = 40
    embedding = trimap.TRIMAP(n_inliers=n_inliers,
                          n_outliers=n_outliers,
                          n_random=n_random).fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c='blue', cmap='Spectral')
    plt.title(title)
    plt.xlabel("age")
    plt.ylabel("serum cholestoral")
    # Добавляем текст с информацией о переменных
    info_text = f'n_inliers: {n_inliers}\nn_outliers: {n_outliers}\nn_random: {n_random}'
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

def visualize_with_umap(scaled_data, title):
    n_neighbors = 60
    n_components = 30
    metric = 'euclidean'
    umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric)  # параметры можно настроить
    embedding = umap_model.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c='blue', cmap='Spectral')
    plt.title(title)
    plt.xlabel("age")
    plt.ylabel("serum cholestoral")
    # Добавляем текст с информацией о переменных
    info_text = f'n_neighbors: {n_neighbors}\nn_components: {n_components}\nmetric: {metric}'
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
def main():
    # Загрузка данных
    # df = pd.read_csv("C:/Users/masma/OneDrive/Документы/GitHub/ISaT_MIREA/practice2/mammoth1.csv")
    #
    # # Удаляем строки с хотя бы одним пропуском
    # df = df.dropna()
    #
    # x_full = df['x'].values
    # print(x_full)
    # y_full = df['y'].values
    # print("------------")
    # print(y_full)

    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    df = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    x_full = df.data.features
    y_full = df.data.targets

    X = x_full.iloc[:, [3, 4]]

    # Убираем ненужные колонки, если есть (например, текстовые данные)
    # numerical_data = df.select_dtypes(include=['float64', 'int64'])

    # Пример уменьшения данных до 10%
    # numerical_data = numerical_data.sample(frac=0.1, random_state=42)

    # Масштабирование данных
    # scalers = {
    #     'MinMax Scaling': MinMaxScaler(),
    #     'Standard Scaling': StandardScaler(),
    #     'Robust Scaling': RobustScaler()
    # }

    scaled_data_MinMaxScaler = MinMaxScaler().fit_transform(X)
    scaled_data_StandardScaler = StandardScaler().fit_transform(X)
    scaled_data_RobustScaler = RobustScaler(quantile_range=(25, 75)).fit_transform(X)

    visualize_with_tsne(scaled_data_MinMaxScaler, f"TSNE with MinMax Scaling")

    visualize_with_tsne(scaled_data_StandardScaler, f"TSNE with Standard Scaling")

    visualize_with_tsne(scaled_data_RobustScaler, f"TSNE with RobustScaler")

    visualize_with_umap(scaled_data_MinMaxScaler, f"UMAP with MinMax Scaling")

    visualize_with_umap(scaled_data_StandardScaler, f"UMAP with Standard Scaling")

    visualize_with_umap(scaled_data_RobustScaler, f"UMAP with RobustScaler")

    visualize_with_trimap(scaled_data_MinMaxScaler, f"TriMAP with MinMax Scaling")

    visualize_with_trimap(scaled_data_StandardScaler, f"TriMAP with Standard Scaling")

    visualize_with_trimap(scaled_data_RobustScaler, f"TriMAP with RobustScaler")

    visualize_with_pacmap(scaled_data_MinMaxScaler, f"PacMAP with MinMax Scaling")

    visualize_with_pacmap(scaled_data_StandardScaler, f"PacMAP with Standard Scaling")

    visualize_with_pacmap(scaled_data_RobustScaler, f"PacMAP with RobustScaler")

    distributions = [
        ("Unscaled data", X),
        ("Data after MinMax Scaling", scaled_data_MinMaxScaler),
        ("Data after Standard Scaling", scaled_data_StandardScaler),
        ("Data after Robust Scaling", scaled_data_RobustScaler),
    ]

    # scale the output between 0 and 1 for the colorbar
    y = minmax_scale(y_full)

    #make_plot(1, distributions, y_full, y)
    #make_plot(2, distributions, y_full, y)
    #make_plot(3, distributions, y_full, y)


def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
        ax_colorbar,
    )

def plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
    cmap = getattr(cm, "plasma_r", cm.hot_r)
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    colors = cmap(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(
        X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
    )
    hist_X1.axis("off")

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(
        X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
    )
    hist_X0.axis("off")

def make_plot(item_idx, distributions, y_full, y):
    cmap = getattr(cm, "plasma_r", cm.hot_r)
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(
        axarr[0],
        X,
        y,
        hist_nbins=200,
        x0_label='x',
        x1_label='y',
        title="Full data",
    )

    # zoom-in
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
        X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1)
    plot_distribution(
        axarr[1],
        X[non_outliers_mask],
        y[non_outliers_mask],
        hist_nbins=50,
        x0_label='x',
        x1_label='y',
        title="Zoom-in",
    )

    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(
        ax_colorbar,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
        label="Color mapping for values of y",
    )

    plt.show()





if __name__ == '__main__':
    main()