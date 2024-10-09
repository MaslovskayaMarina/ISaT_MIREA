import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

    # Чтение данных из CSV
    df = pd.read_csv("/home/berkunov/Documents/GitHub/ISaT_MIREA/practice2/mammoth2.csv")

    df = df.dropna()
    df.x.value_counts()

    sns.pairplot(df.drop("x", axis=1), hue='y')
    plt.show()


if __name__ == '__main__':
    main()
    print('loh')