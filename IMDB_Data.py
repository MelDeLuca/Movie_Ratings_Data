# Genres and star ratings
import pandas as pd
import seaborn as sns

imdbdata = "https://bit.ly/imdbratings"
imdb = pd.read_csv(imdbdata, usecols = ["star_rating", "genre"])
print(imdb.info())

# No missing values; may proceed
print()
genrelist = []
for entry in imdb["genre"]:
    if entry not in genrelist:
        genrelist.append(entry)
print(genrelist, " so there are ", len(genrelist), " genres.")

print()
ntimesarray = []
for genrename in genrelist:
    numberoftimes = (imdb["genre"] == genrename).sum()
    ntimesarray.append(numberoftimes)
    print(genrename, " occurs ", numberoftimes, "time(s).")

print()
print("Here's a corresponding dictionary: ")
genredict = dict(zip(genrelist, ntimesarray))
print(genredict)

print()
sns.set_style("ticks")
sns.lmplot(x = "genre", y = "star_rating", data = imdb, fit_reg = False, height = 6, aspect = 3, hue = "genre")
