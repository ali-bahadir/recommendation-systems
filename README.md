
# Recommendation Systems

This repository contains projects focused on building and evaluating recommendation systems using different methodologies. The projects include:

1. **Association Rule-Based Recommender System**: Uses Association Rule Learning to recommend services based on user behavior patterns.
2. **Hybrid Recommender System**: Combines user-based and item-based collaborative filtering methods to recommend movies to users.


## Business Problems

### 1. Association Rule-Based Recommender System
Armut, Turkey's largest online service platform, connects service providers with customers. The goal of this project is to recommend services to users based on their past service purchases using Association Rule Learning.

### 2. Hybrid Recommender System
Using the MovieLens dataset, this project aims to recommend 10 movies to a user by combining user-based and item-based collaborative filtering methods.

---

## Datasets

### Association Rule-Based Recommender System
The dataset includes services purchased by users on Armut, along with service categories and purchase dates.

- **UserId**: Unique customer ID
- **ServiceId**: Anonymized service ID under different categories
- **CategoryId**: Anonymized category ID (e.g., cleaning, repair)
- **CreateDate**: Purchase date of the service

### Hybrid Recommender System
The dataset is sourced from MovieLens and contains information about movies, user ratings, and timestamps.

- **movie.csv**: Contains movie titles and genres
  - **movieId**: Unique movie ID
  - **title**: Movie title
  - **genres**: Genre of the movie

- **rating.csv**: Contains user ratings
  - **userId**: Unique user ID
  - **movieId**: Unique movie ID
  - **rating**: User rating for the movie
  - **timestamp**: Rating date

---

## Projects

### 1. Association Rule-Based Recommender System
**Steps**:
- **Data Preparation**:
  - Combine `ServiceId` and `CategoryId` to create a new service identifier.
  - Create unique basket IDs for each user's monthly service purchases.
- **Association Rule Learning**:
  - Generate association rules for the prepared dataset.
  - Recommend services to users based on the most recent service they purchased.

### 2. Hybrid Recommender System
**Steps**:
- **Data Preparation**:
  - Filter movies with fewer than 1,000 ratings.
  - Create a user-movie rating matrix.
- **User-Based Recommendation**:
  - Calculate similarities between users.
  - Recommend movies based on similar users' preferences.
- **Item-Based Recommendation**:
  - Identify the most similar movies to the ones rated highly by the user.
- **Hybrid Approach**:
  - Combine user-based and item-based recommendations for a comprehensive output.

