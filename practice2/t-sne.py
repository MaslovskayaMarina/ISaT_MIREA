# Задание 3-4. Визуализация в двухмерном пространстве набора
# данных с использованием алгоритмов нелинейного
# снижения размерности t-sne, UMAP, TriMap и PaCMAP алгоритмов. Дедлайн на занятиях: 13 октября.

# Задание 5-6. Разработка knn-, SVM-, RF-классификаторов. Изобразить средствами Python
# 2 дерева в RF. Дедлайн на занятиях: 27 октября.

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    df = pd.read_csv("/home/berkunov/Documents/GitHub/ISaT_MIREA/practice2/mammoth.csv")

    print(df)
    pass


if __name__ == '__main__':
    main()