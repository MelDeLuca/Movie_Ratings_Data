# Predicting star ratings

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

rs = 4

movie_file = "https://bit.ly/imdbratings"
movie_data = pd.read_csv(movie_file)

print(movie_data.columns)
the_movie_data = pd.DataFrame(movie_data, columns = ["star_rating","content_rating","genre","duration"])
print(movie_data.info(), movie_data.describe(), movie_data.head())

# Nothing is empty, so no need to impute (which would probably be with a mean or median, depending on sd) 
# Might need to remove outliers 

# Keep star_rating, content_rating, duration; omit title, genre, and actors_list
# Am not going to use title or actor names in the model; tried to use genre,
# but with so many smaller sub-genres, it wasn't easy to group them together
# Does horror go with mystery or sci-fiction? Putting them all together isn't
# much better, as mystery and sci-fiction are fairly different
# It was better to remove genre; maybe another project 

movie_data = movie_data.drop(["title", "genre", "actors_list"], axis = 1)
print(movie_data)

# Star rating, being from a top movie list, has relatively small sd, so keep all

content_list = []
for c_entry in movie_data["content_rating"]:
    if c_entry not in content_list:
        content_list.append(c_entry)
print(content_list)


remove_content = []
for c_rating in content_list:
    letter_rating = (movie_data["content_rating"] == c_rating).sum()
    print(c_rating, ":", letter_rating)
    
# Might be unconventional, but since the rating systems have changed over time,
#    will reclassify into subgroups based off of what a generic modern-day rating would be
# For PASSED and APPROVED, a quick glimpse shows the former is largely PG
#    and the latter leans toward PG-13; note both have exceptions 

# Maybe one-hot encoding?
print()
print("Rating scale: 4 is restricted to adults, 3 is okay for older kids/teens, 2 for everybody, 1 is unknown.") 
    
# 4 is Mature: R, X, TV-MA, NC-17
# 3 No young kids, but not only adults (Teens): PG-13, GP, APPROVED 
# 2 Kid-Friendly, young and/or old kids: PG, G, PASSED
# 1 No rating: NOT RATED, UNRATED    

rating_list = ["R", "X", "TV-MA", "NC-17", "PG-13", "GP", "APPROVED", "PG", "G", "PASSED", "NOT RATED", "UNRATED"]    
for the_movie_rating in movie_data["content_rating"]:
    for rating in rating_list: 
        if the_movie_rating in rating_list[0:4]:
            movie_data["content_rating"] = movie_data["content_rating"].replace(the_movie_rating, 4)
        elif the_movie_rating in rating_list[4:7]:
            movie_data["content_rating"] = movie_data["content_rating"].replace(the_movie_rating, 3)
        elif the_movie_rating in rating_list[7:10]:
            movie_data["content_rating"] = movie_data["content_rating"].replace(the_movie_rating, 2)
        else:
            movie_data["content_rating"] = movie_data["content_rating"].replace(the_movie_rating, 1)

# Don't need loc or iloc- for now
    
# Onward to duration

print(max(movie_data.duration)-min(movie_data.duration))

for moviemin in movie_data["duration"]:
    if moviemin >= 180:
        longm = (movie_data["duration"] >= 180).sum()
    elif moviemin >= 90:
        medm = (movie_data["duration"] >=90).sum()
    else: 
        shortm = (movie_data["duration"] < 90).sum()

print("Movie length categories: <90 min.: ", shortm, "; 90-179: ", medm, "; 180+: ", longm)

# Will leave duration; each interval has enough

print(movie_data.info(), movie_data.describe())
 
print(movie_data)

# Want to predict star rating
y = movie_data.star_rating

movie_descriptors = ["content_rating", "duration"]
X = movie_data[movie_descriptors]

# Try decision tree

dt_movie_model = DecisionTreeRegressor(random_state = rs)    
dt_movie_model.fit(X,y)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = rs)
dt_movie_model.fit(train_X, train_y)
dt_val_predictions = dt_movie_model.predict(val_X)
print("The first Decision Tree MAE is: ", mean_absolute_error(val_y, dt_val_predictions))

# The MAE is about .3; good that it's below the sd
# Improve this with leaf nodes!

def getthemae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    dt_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = rs)
    dt_model.fit(train_X, train_y)
    prediction_val = dt_model.predict(val_X)
    themae = mean_absolute_error(val_y, prediction_val)
    return themae

leastmae=979
bestmln = 979 

for max_leaf_nodes in range(2, 979, 2):
    mae = getthemae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    if mae < leastmae:
        leastmae = mae
        bestmln = max_leaf_nodes
print("Best MLN: ", bestmln, " has MAE ", leastmae)
print()

dt_movie_model = DecisionTreeRegressor(max_leaf_nodes = bestmln, random_state = rs)
dt_movie_model.fit(train_X, train_y)
val_dt_preds = dt_movie_model.predict(val_X)
print("The better MAE for Decision Tree is now: ", mean_absolute_error(val_y, val_dt_preds))

# Much better!

errors = abs(val_y-val_dt_preds)
mape = 100*errors/val_y
dt_accuracy = 100 - np.mean(mape)
print(round(dt_accuracy, 2), "% is the accuracy.")
print()

# How about RandomForest?

rf_movie_model = RandomForestRegressor(n_estimators = 500, random_state = rs)
rf_movie_model.fit(X,y)
rf_movie_model.fit(train_X, train_y)
rf_val_predictions = rf_movie_model.predict(val_X) 
print("The Random Forest MAE is: ", mean_absolute_error(val_y, rf_val_predictions))

#So using better parameters for DT is better than a straight-out RF

# Now RF accuracy:

errors = abs(val_y-rf_val_predictions)
mape = 100*errors/val_y
rf_accuracy = 100-np.mean(mape)
print(round(rf_accuracy,2), "% is the accuracy.")
