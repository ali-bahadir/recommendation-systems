#############################################
# PROJECT: Hybrid Recommendation System
#############################################

# For the given user ID, make predictions using item-based and user-based recommendation methods.
# Take 5 recommendations from the user-based model and 5 from the item-based model,
# and combine them to create a final list of 10 recommendations.

#############################################
# Task 1: Data Preparation
#############################################

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Step 1: Read the Movie and Rating datasets.

movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')
movie.head()

rating = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/rating.csv')
rating.head()

# Step 2: Add the names and genres of movies to the Rating dataset using the Movie dataset.
# The Rating dataset only contains the movie IDs that users have rated.
# Add the names and genres for these IDs from the Movie dataset.

df = movie.merge(rating, how="left", on="movieId")  # Common column is "movieId"
df.head()
df.shape

# Step 3: Calculate how many users have rated each movie.
# Remove movies with fewer than 1000 ratings from the dataset.

comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts

rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

# Step 4: Create a pivot table where rows are user IDs, columns are movie titles,
# and values are the ratings given by users.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")


# Step 5: Turn the above steps into a function.

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()

#############################################
# Task 2: Identifying Movies Watched by the User
#############################################

# Step 1: Select a random user ID.

random_user = 108170

# Step 2: Create a new dataframe, random_user_df, containing the data for the selected user.

random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape

# Step 3: Store the movies rated by the selected user in a list called movies_watched.

movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
# If a movie is rated, it is watched. Keep non-NaN values.

#############################################
# Task 3: Access Data and IDs of Other Users Watching the Same Movies
#############################################

# Step 1: Select the columns corresponding to the movies watched by the user from user_movie_df,
# and create a new dataframe called movies_watched_df.

movies_watched_df = user_movie_df[movies_watched]

# Step 2: Create a new dataframe, user_movie_count, containing the count of the movies watched
# by each user among the movies watched by the selected user.

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# Step 3: Identify users who have watched at least 60% of the movies rated by the selected user.
# Create a list of their IDs called users_same_movies.

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

#############################################
# Task 4: Identifying the Most Similar Users to the Selected User
#############################################

# Step 1: Filter movies_watched_df to include only users in the users_same_movies list.

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()

# Step 2: Calculate the correlations between users and create a new dataframe called corr_df.

corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()

# Step 3: Filter users with a correlation higher than 0.65 with the selected user,
# and create a new dataframe called top_users.

top_users = (corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].
             reset_index(drop=True))
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()

# Step 4: Merge the top_users dataframe with the Rating dataset.

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()

#############################################
# Task 5: Calculating Weighted Average Recommendation Scores
#############################################

# Step 1: Create a new variable, weighted_rating, by multiplying the corr and rating values for each user.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Step 2: Create a new dataframe, recommendation_df, containing the average weighted rating for each movie.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"}).reset_index()
recommendation_df.head()

# Step 3: Filter movies with a weighted rating greater than 3.5, sort them,
# and save the top 5 movies as movies_to_be_recommend.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating",
                                                                                                   ascending=False)

# Step 4: Retrieve the names of the top 5 recommended movies.

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

#############################################
# Step 6: Item-Based Recommendation
#############################################

# Recommend movies based on the most recently watched and highest-rated movie of the user.
user = 108170

# Step 1: Read the Movie and Rating datasets.

movie = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Recommendation Systems/datasets/movie_lens_dataset/rating.csv')

# Step 2: Identify the most recently rated movie with a rating of 5 by the user.

movie_id = rating[(rating["userId"] == user) &
                  (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Step 3: Filter user_movie_df based on the selected movie ID.

movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

# Step 4: Calculate the correlation of the selected movie with other movies and sort the results.

user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)


# Function for the last two steps
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


# Step 5: Recommend the top 5 movies (excluding the selected movie itself).

movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
movies_from_item_based[1:6].index
