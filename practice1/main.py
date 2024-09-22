# 1. Применить для тестовых вариантов и вариантов из репозиториев различные алгоритмы поиска ассоциативных правил при одинаковых начальных
# условиях (при одинаковых пороговых значениях для поддержки и достоверности) и сравнить полученные результаты. Для тестовых вариантов выполнить
# ручные расчеты (например, с применением MS Excel) и расчеты с применением
# программных библиотек на языке Python. Для вариантов из репозиториев выполнить расчеты с применением программных библиотек на языке Python.
# В качестве алгоритмов поиска ассоциативных правил использовать алгоритмы:
#   Apriori (https://pypi.org/project/apriori-python/);
#   Efficient Apriori (https://pypi.org/project/efficient-apriori/);
#   FPGrowth (https://pypi.org/project/fpgrowth-py/).
# 2. Сформировать базы ассоциативных правил с уровнем минимальной достоверности 60% и 80%. Вычислить для ассоциативных правил поддержку, достоверность, значимость.
# 3. Оценить время формирования искомых ассоциативных правил с применением различных алгоритмов и построить диаграммы, позволяющие выполнить сравнительный анализ.
# 4. Выполнить визуализацию ассоциативных правил (https://pypi.org/project/pyarmviz/).
#
# Dataset: csv where { Date, Time, Transaction, Item }
import csv
from apriori_python import apriori
import pandas as pd
from efficient_apriori import apriori as eff_apriori
from fpgrowth_py import fpgrowth


def data_generator(filepath):
    def data_gen():
        with open(filepath) as file:
            for line in file:
                yield tuple(k.strip() for k in line.split(','))

    return data_gen


def main():
    df = pd.read_csv("/home/berkunov/Documents/GitHub/ISaT_MIREA/practice1/BreadBasket_DMS.csv")
    df = df.groupby('Transaction')['Item'].apply(list).tolist()

    # apriori_python
    # freq_item_set, rules = apriori(df, minSup=0.01, minConf=0.01)
    # print(rules)

    # efficient_apriori
    # freq_item_set,  rules = eff_apriori(df, min_support=0.01, min_confidence=0.01)
    # rules_rhs = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules)
    # for rule in sorted(rules_rhs, key = lambda rule: rule.confidence):
    #     print(rule)
    # print(rules)

    # fpgrowth
    freq_item_set, rules = fpgrowth(df, minSupRatio=0.05, minConf=0.05)
    print(rules)



if __name__ == '__main__':
    main()
