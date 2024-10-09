# Задание 3-4. Визуализация в двухмерном пространстве набора
# данных с использованием алгоритмов нелинейного
# снижения размерности t-sne, UMAP, TriMap и PaCMAP алгоритмов. Дедлайн на занятиях: 13 октября.

# Задание 5-6. Разработка knn-, SVM-, RF-классификаторов. Изобразить средствами Python
# 2 дерева в RF. Дедлайн на занятиях: 27 октября.

# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.manifold import TSNE
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize_with_tsne(scaled_data, title):
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, max_iter=500, random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='blue', cmap='Spectral')
    plt.title(title)
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.show()


def main():
    # Загрузка данных
    df = pd.read_csv("/home/berkunov/Documents/GitHub/ISaT_MIREA/practice2/mammoth.csv")

    # Убираем ненужные колонки, если есть (например, текстовые данные)
    numerical_data = df.select_dtypes(include=['float64', 'int64'])

    # Пример уменьшения данных до 10%
    numerical_data = numerical_data.sample(frac=0.1, random_state=42)

    # Масштабирование данных
    scalers = {
        'MinMax Scaling': MinMaxScaler(),
        'Standard Scaling': StandardScaler(),
        'Robust Scaling': RobustScaler()
    }

    # Применяем t-SNE и визуализируем для каждого метода масштабирования
    for scaler_name, scaler in scalers.items():
        scaled_data = scaler.fit_transform(numerical_data)
        visualize_with_tsne(scaled_data, f"t-SNE with {scaler_name}")


if __name__ == '__main__':
    main()
    print('end')