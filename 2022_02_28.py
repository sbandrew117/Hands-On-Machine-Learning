
from ast import increment_lineno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl





def practice():
    
    housing = pd.read_csv('C:\\Users\\kjsj9\\Desktop\\handson-ml-master\\handson-ml-master\\datasets\\housing\\housing.csv')

    print(housing["ocean_proximity"].value_counts())
    
    print(housing.describe())

def split_train_set(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

    





if __name__ == "__main__":
    practice()