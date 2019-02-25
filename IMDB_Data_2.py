# Duration and star ratings
# Improve Decision Tree with max leaf nodes

import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

rs = 4

imdb_data = "https://bit.ly/imdbratings"
imdb = pd.read_csv(imdb_data)
print(imdb.info())
print(imdb.describe())

# No missing values; may proceed

print(imdb.columns)

# Want to predict: star_rating
y = imdb.star_rating

imdb_features = ['duration']
X = imdb[imdb_features]

imdb_model = DecisionTreeRegressor(random_state = rs)
imdb_model.fit(X,y)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = rs)
imdb_model.fit(train_X, train_y) 
val_predictions = imdb_model.predict(val_X)
print()
# Validation data versus the predictions
print("The MAE is: ", mean_absolute_error(val_y, val_predictions))

def getthemae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = rs)
    model.fit(train_X, train_y)
    prediction_val = model.predict(val_X)
    themae = mean_absolute_error(val_y, prediction_val)
    return themae

print()
leastmae=979
bestmln = 950 

for max_leaf_nodes in range(50, 979, 50):
    mae = getthemae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max Leaf Nodes: ", max_leaf_nodes, " Mean Absolute Error: ", mae)
    if mae < leastmae:
        leastmae = mae
        bestmln = max_leaf_nodes
print()

print("Best MLN: ", bestmln, " has MAE ", leastmae)
imdb_dt_model = DecisionTreeRegressor(max_leaf_nodes = bestmln, random_state = rs)
imdb_dt_model.fit(train_X, train_y)
val_dt_preds = imdb_dt_model.predict(val_X)
print("The better MAE is now: ", mean_absolute_error(val_y, val_dt_preds))