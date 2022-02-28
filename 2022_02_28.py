
import os
from posixpath import split
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score



# def practice():
    
housing = pd.read_csv('C:\\Users\\kjsj9\\Desktop\\handson-ml-master\\handson-ml-master\\datasets\\housing\\housing.csv')

print("\nocean proximity:\n", housing["ocean_proximity"].value_counts())

print(housing.describe())

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")
    
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
print(housing["income_cat"].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)

housing = strat_train_set.copy()
print(housing.plot(kind = "scatter", x = "longitude", y = "latitude"))
print(housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1))

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4,
                s = housing["population"]/100, label = "population", figsize = (10, 7),
                c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True, sharex = False)

print(plt.legend())

corr_matrix = housing.corr()
print("\ncorr_matrix:\n", corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms",
                "housing_median_age"]
print(scatter_matrix(housing[attributes], figsize = (12, 8)))

#median_house_value & medina_income correlation
print(housing.plot(kind = "scatter", x = "median_income", y = "median_house_value",
                    alpha = 0.1))

#creating more columns(to search for more correlation)
housing["room_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing ["households"]

corr_matrix = housing.corr()
print("\nnew corr_matrix:\n", corr_matrix["median_house_value"].sort_values(ascending = False))

housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()

#total_bedrooms column 특성에 값이 없는 경우가 있어 고치는 방법:

housing.dropna(subset = ["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis = 1) # option 2
median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace = True)

imputer = SimpleImputer(strategy = "median")

#SimpleImputer only works on number-based data
#since ocean_proximity's dtype is object, drop ocean_proximity to use imputer

housing_num = housing.drop("ocean_proximity", axis = 1)

imputer.fit(housing_num)

print("\nimputer statistics:\n", imputer.statistics_)
print("\nhousing_num medians:\n", housing_num.median().values)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns = housing_num.columns,
                            index = list(housing.index.values))

#factorize()-> 각 카테고리를 다른 정숫값으로 매핑해줌

housing_cat = housing["ocean_proximity"]
print("\nhousing_cat head 10:\n", housing_cat)
housing_cat_encoded, housing_categories = housing_cat.factorize()
print("\nfactorized ocean_proximity:\n", housing_cat_encoded[:10])

encoder = OneHotEncoder(categories = 'auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
print("\n encoded housing_cat as 1hot:\n", housing_cat_1hot)

print("\n encoded housing_cat as 1hot to array:\n", housing_cat_1hot.toarray())

cat_encoder = OrdinalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
print("\nhousing_cat_1hot:\n", housing_cat_1hot)

#categoricalencoder error
'''
    cat_encoder = CategoricalEncoder(encoding = "onehot-dense")
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
    print(housing_cat_1hot)
   ''' 

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)

   
   
   
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),])

housing_num_tr = num_pipeline.fit_transform(housing_num)




class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy = "median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])   

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OrdinalEncoder())    
])


full_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
print("\nhousing prepared:\n", housing_prepared)

print("\nhousing prepared shape:\n", housing_prepared.shape)


#훈련 세트에서 훈련하고 평가하기

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("예측", lin_reg.predict(some_data_prepared))
print("레이블", list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("\nRMSE:\n", lin_rmse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

print("\ntree RMSE:\n", tree_rmse)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring = "neg_mean_squared_error", cv = 10) #10번 훈련

tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("\nScores:\n", scores)
    print("\nMean:\n", scores.mean())
    print("\nStandard deviation:\n", scores.std())
    
if __name__ == "__main__":
    display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring = "neg_mean_squared_error", cv = 10)
