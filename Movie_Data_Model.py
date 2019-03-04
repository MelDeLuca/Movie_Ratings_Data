# Use IMDB data to predict star ratings


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# Setting the random_state to 4 for uniformity; can be changed if desired 
rs = 4

movie_file = "https://bit.ly/imdbratings"
movie_data = pd.read_csv(movie_file)

print(movie_data.columns)


# Before removing any features, check if anything is missing or if there are any outliers
the_movie_data = pd.DataFrame(movie_data, columns = ["star_rating","content_rating","genre","duration"])
print(movie_data.info(), "\n", movie_data.describe(), "\n", movie_data.head())


# Nothing is empty, so no need to impute with mean; so far, no need to remove outliers 

# Features: Keep star_rating, content_rating, duration; omit title, genre, and actors_list

# Will not use title or actor names in the model; tried to use genre, but with 
# so many smaller sub-genres, it wasn't easy to group them together
# Does horror go with mystery or sci-fiction? Putting them all together isn't
# much better, as mystery and sci-fiction are fairly different.
# So will just omit genre.

movie_data = movie_data.drop(["title", "genre", "actors_list"], axis = 1)
print(movie_data)


# Star rating, being from a top movie list, has relatively small sd (about a third of a star), so keep all

content_list = []
for c_entry in movie_data["content_rating"]:
    if c_entry not in content_list:
        content_list.append(c_entry)
print(content_list)

for c_rating in content_list:
    letter_rating = (movie_data["content_rating"] == c_rating).sum()
    print(c_rating, ":", letter_rating)


# Might be unconventional, but since the rating systems have changed over time,
# will reclassify into subgroups based off of what a generic modern-day rating would be.
# For PASSED and APPROVED, a quick glimpse shows the former is largely PG
# and the latter leans toward PG-13, as the former is earlier and thus likely more affected
# by the Hayes Code ; note both categories have exceptions, and this can affect the model's accuracy 

print()
print("Rating scale: 4 is restricted to adults, 3 is okay for older kids/teens, 2 for everybody, 1 is unknown.")


# 4 is Mature: R, X, TV-MA, NC-17
# 3 is No young kids, but not only adults (Teens): PG-13, GP, APPROVED 
# 2 is Kid-Friendly, young and/or old kids: PG, G, PASSED
# 1 is No rating: NOT RATED, UNRATED    

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

print(max(movie_data.duration)-min(movie_data.duration), " minutes between shortest and longest movie")

# About a three hour difference in range, half hour for standard deviation (from before)


for moviemin in movie_data["duration"]:
    if moviemin >= 180:
        longm = (movie_data["duration"] >= 180).sum()
    elif moviemin >= 90:
        medm = (movie_data["duration"] >=90).sum()
    else: 
        shortm = (movie_data["duration"] < 90).sum()

print("Movie length categories:", "\n", " <90 min.: ", shortm, "\n", " 90-179: ", medm, "\n", " 180+: ", longm)

# Will leave duration; each interval has enough to not be heavily swayed by unusual outliers, 
# though really short and really long movies are smaller subgroups

print(movie_data.info(), movie_data.describe())
print()
print(movie_data)


# Now use this data to predict star rating

y = movie_data.star_rating

movie_descriptors = ["content_rating", "duration"]
X = movie_data[movie_descriptors]

# Try decision tree

dt_movie_model = DecisionTreeRegressor(random_state = rs)    

# dt_movie_model.fit(X,y)
# Note that X refers to features, y refers to labels, train to training data, 
# val to validation/test data (different, yes, but here will treat as same.)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = rs)
dt_movie_model.fit(train_X, train_y)
dt_val_predictions = dt_movie_model.predict(val_X)
print("The first Decision Tree MAE is: ", mean_absolute_error(val_y, dt_val_predictions))


# See if this can be improved with setting max leaf nodes

def get_the_new_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    dt_revised_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = rs)
    dt_revised_model.fit(train_X, train_y)
    prediction_val = dt_revised_model.predict(val_X)
    the_new_mae = mean_absolute_error(val_y, prediction_val)
    return the_new_mae

least_mae = 979
best_mln = 979 

for max_leaf_nodes in range(2, 979, 2):
    mae = get_the_new_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    if mae < least_mae:
        least_mae = mae
        best_mln = max_leaf_nodes
print("Best MLN (MAX LEAF NODES): ", best_mln, " has MAE (MEAN ABSOLUTE ERROR):", least_mae)
print()

dt_revised_movie_model = DecisionTreeRegressor(max_leaf_nodes = best_mln, random_state = rs)
dt_revised_movie_model.fit(train_X, train_y)
dt_val_predictions = dt_revised_movie_model.predict(val_X)
print("The better MAE for Decision Tree is now: ", mean_absolute_error(val_y, dt_val_predictions))


# Much better!
# The MAE is far enough from 0 to dispel ideas of serious overfitting
# Instead of accuracy_score, will use mean absolute percentage error; no absolute precision error

errors = abs(val_y-dt_val_predictions)
mape = 100*errors/val_y
dt_accuracy = 100 - np.mean(mape)
print(round(dt_accuracy, 2), "% is the accuracy.")


# How about RandomForest?

# Pick 500 trees
# More had slowed run time and only increased accuracy by .01, less was about .01 less
# and not noticeably faster

rf_movie_model = RandomForestRegressor(n_estimators = 500, random_state = rs)

#rf_movie_model.fit(X,y)

rf_movie_model.fit(train_X, train_y)
rf_val_predictions = rf_movie_model.predict(val_X) 
print("The Random Forest MAE is: ", mean_absolute_error(val_y, rf_val_predictions))

# So, at least here, using better parameters for DT is better than a basic RF

# Now RF accuracy:

errors = abs(val_y-rf_val_predictions)
mape = 100*errors/val_y
rf_accuracy = 100-np.mean(mape)
print(round(rf_accuracy,2), "% is the accuracy.")


if rf_accuracy == dt_accuracy:
    better_model = "they're the same. Neither"
elif rf_accuracy > dt_accuracy:
    better_model = "random forest"
else:
    better_model = "decision tree"

print("Between decision tree and random forest, {} is the more accurate model.".format(better_model))


# Since so few features, the decision trees are relatively stable to begin with, 
# explaining the slightly higher accuracy
# I would suspect using all the features and, tuning the parameters, could make 
# RF the better model
