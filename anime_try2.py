from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import operator
import numpy as np
import seaborn as sns
import statistics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load datasets
anime_data = pd.read_csv("anime.csv")  
ratings_data = pd.read_csv("rating.csv")  
print(anime_data.head(), "\n")
anime_data.dataframeName = 'anime.csv'

#------------------------------------anime.csv analysis------------------------------
#Descriptive Statistics
print(anime_data[["rating", "members"]].describe(), "\n")

#Get Show Names
unique_names = anime_data["name"].unique()
print(unique_names[:100],"\n")

nRow, nCol = anime_data.shape

#Data Visualization
def plotPerColumnDistribution(df, nGraphShown=10, nGraphPerRow=5):
    nunique = df.nunique()
    # Filter columns with unique values between 1 and 50 for visualization purposes
    filtered_df = df[[col for col in df if 1 < nunique[col] < 50]]
    nRow, nCol = filtered_df.shape
    columnNames = list(filtered_df)
    nGraphRow = int(np.ceil(nCol / nGraphPerRow))  # Calculate the number of rows required for subplots
    
    plt.figure(figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = filtered_df.iloc[:, i]
        if columnDf.dtype == 'object' or columnDf.dtype.name == 'category':  # Check for categorical columns
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:  # Numeric columns
            columnDf.hist()
        plt.ylabel('Counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()


plotPerColumnDistribution(anime_data, 10, 5)

# Grouped descriptive statistics
# Quantile function
def q25(x):

    return x.quantile(0.25)

def q75(x):

    return x.quantile(0.75)



#Descriptive statistic for type and rating
print(anime_data.groupby('type')["rating"] \
    .aggregate(['mean', 'std', 'min', q25, 'median', q75, 'max']) \
    .transpose(),"\n")


#----------------------------rating.csv analysis/cleaning--------------------------------
print(ratings_data.head(),"\n")
ratings = ratings_data[ratings_data["rating"] != -1]
print("After Cleaning:")
print(ratings.groupby(["rating"])["rating"].count(), "\n")

#---------------------------Summary of both--------------------------------------------------

print("There are", str(len(ratings)), "ratings posted.")

# number of unique users
print("A total of", str(len(ratings['user_id'].unique())), "unique user ids.")

# number of unique animes (in anime list, not ratings)
print("A rate total of", str(len(anime_data['anime_id'].unique())), "different animes.")

# avg number of anime rated per user
ratings_per_user = ratings.groupby('user_id')['rating'].count()
print("Mean of ratings per user:", statistics.mean(ratings_per_user.tolist()))

## distribution of ratings posted per user
#_ = plt.title('Distribution of Ratings Posted per User')
#_ = sns.histplot(ratings_per_user)
#plt.show()
#
# avg number of ratings given per anime
ratings_per_anime = ratings.groupby('anime_id')['rating'].count()
print("Mean of ratings per anime:", statistics.mean(ratings_per_anime.tolist()), "\n")

#------------------------------- Cleaning the Data Further----------------------------------
# counts of ratings per anime as a df
ratings_per_anime_df = pd.DataFrame(ratings_per_anime)

# remove if less than mean rounded to the nearest ten.
filtered_rating_per_anime_df = ratings_per_anime_df[ratings_per_anime_df['rating'] >= 1000]

# build a list of anime_ids to keep
popular_anime = filtered_rating_per_anime_df.index.tolist()
print("Length of popular animes:", len(popular_anime), "\n")

# counts ratings per user as a df
ratings_per_user_df = pd.DataFrame(ratings_per_user)

# remove rating if less than double the mean rounded to the nearest hundred
filtered_rating_per_user_df = ratings_per_user_df[ratings_per_user_df['rating'] >= 95]

# build a list of user_ids to keep
profilic_users = filtered_rating_per_user_df.index.tolist()
print("Length of prolific users:", len(profilic_users), "\n")

#anime and users not in this list
filtered_ratings = ratings[ratings.anime_id.isin(popular_anime) & ratings.user_id.isin(profilic_users)]
print("Length of frequent users and popular shows:", len(filtered_ratings), "\n")

# distribution of ratings per anime
_ = plt.title('Distribution of Ratings Posted per Anime')
_ = sns.histplot(filtered_rating_per_anime_df)
plt.show()

# distribution of ratings posted per user
_ = plt.title('Distribution of Ratings Posted per User')
_ = sns.histplot(ratings_per_user)
plt.show()

#--------------------Histogram for Anime and User Ratings--------------------------------------
#palette = sns.color_palette("muted")
#top_anime_temp2 = anime_data.sort_values(["rating"], ascending=False)
#fulldata = ratings.rename(columns={"rating": "user_rating"})
#
#_, axs = plt.subplots(2, 1, figsize=(20, 16), sharex=False, sharey=False)
#plt.tight_layout(pad=6.0)
#
#sns.histplot(top_anime_temp2["rating"], color=palette[1], kde=True, ax=axs[0], bins=20, alpha=1, fill=True, edgecolor=palette[2])
#axs[0].lines[0].set_color(palette[2])
#axs[0].set_title("\nAnime's Average Ratings Distribution\n", fontsize=25)
#axs[0].set_xlabel("Rating\n", fontsize=20)
#axs[0].set_ylabel("Total", fontsize=20)
#
#sns.histplot(fulldata["user_rating"], color=palette[2], kde=True, ax=axs[1], bins="auto", alpha=1, fill=True)
#axs[1].lines[0].set_color(palette[1])
#axs[1].set_title("\n\n\nUsers Anime Ratings Distribution\n", fontsize=25)
#axs[1].set_xlabel("Rating", fontsize=20)
#axs[1].set_ylabel("Total", fontsize=20)
#
#sns.despine(left=True, bottom=True)
#plt.show()

#-------------------------------------------rating matrix between users and anime------------------------------------------------
rating_matrix = filtered_ratings.pivot_table(index='user_id', columns='anime_id', values='rating')

# replace NaN values with 0
rating_matrix = rating_matrix.fillna(0)

# display the top few rows
print("Rating Matrix between anime_id and user_id")
print(rating_matrix.iloc[:15, :15].to_string(), "\n")

# Create training and test sets
train_data, test_data = train_test_split(filtered_ratings, test_size=0.5, random_state=42)
rating_matrix_train = train_data.pivot_table(index='user_id', columns='anime_id', values='rating').fillna(0)
rating_matrix_test = test_data.pivot_table(index='user_id', columns='anime_id', values='rating').fillna(0)

# Build content similarity matrix
anime_features = anime_data[['anime_id', 'rating', 'members']].dropna()
content_similarity_matrix = pd.DataFrame(
    cosine_similarity(anime_features[['rating', 'members']]),
    index=anime_features['anime_id'],
    columns=anime_features['anime_id']
)

# Similar Users Function
def similar_users(user_id, matrix, k=5):
    user = matrix[matrix.index == user_id]
    other_users = matrix[matrix.index != user_id]
    similarities = cosine_similarity(user, other_users)[0].tolist()
    indices = other_users.index.tolist()
    index_similarity = dict(zip(indices, similarities))
    sorted_users = sorted(index_similarity.items(), key=operator.itemgetter(1), reverse=True)[:k]
    return [u[0] for u in sorted_users]

def display_top_watched_shows(user_id, similar_user_indices, matrix, anime_data, top_n=10):
   
    # User's watched anime
    user_ratings = matrix.loc[user_id]
    user_watched = user_ratings[user_ratings > 0]
    
    # Anime watched by similar users
    similar_users_ratings = matrix.loc[similar_user_indices]
    similar_users_mean_ratings = similar_users_ratings.mean(axis=0)  # Average ratings across similar users
    
    # Find overlap between user's watched anime and similar users' ratings
    overlap_anime_ids = user_watched.index.intersection(similar_users_mean_ratings.index)
    
    # Combine ratings and sort by the mean of similar users' ratings
    overlap_anime_info = anime_data[anime_data['anime_id'].isin(overlap_anime_ids)]
    overlap_anime_info = overlap_anime_info.copy()
    overlap_anime_info.loc[:, 'similar_users_mean_rating'] = overlap_anime_info['anime_id'].map(similar_users_mean_ratings)
    overlap_anime_info = overlap_anime_info.sort_values(by='similar_users_mean_rating', ascending=False)
    
    # Display top N results
    print(f"\nTop {top_n} Rated Shows Watched by Both User {user_id} and Similar Users:")
    print(overlap_anime_info[['anime_id', 'name', 'rating', ]].head(top_n))





# Hybrid Recommendation Function
def hybrid_recommendation(user_index, similar_user_indices, matrix, content_similarity_matrix, anime_data, items=5, alpha=0.5):
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    collaborative_scores = similar_users.mean(axis=0)

    # Get seen and unseen anime IDs for the current user
    user_ratings = matrix.loc[user_index]
    seen_anime_ids = user_ratings[user_ratings > 0].index.tolist()
    unseen_anime_ids = [anime_id for anime_id in anime_data['anime_id'] if anime_id not in seen_anime_ids]

    # Filter unseen_anime_ids and seen_anime_ids to match content_similarity_matrix
    unseen_anime_ids = [anime_id for anime_id in unseen_anime_ids if anime_id in content_similarity_matrix.index]
    seen_anime_ids = [anime_id for anime_id in seen_anime_ids if anime_id in content_similarity_matrix.columns]

    # Calculate content scores
    if seen_anime_ids:
        content_scores = content_similarity_matrix.loc[unseen_anime_ids, seen_anime_ids].mean(axis=1)
    else:
        content_scores = pd.Series(0, index=unseen_anime_ids)  # Default to 0 if no seen items

    # Align collaborative_scores with unseen_anime_ids
    collaborative_scores = collaborative_scores.reindex(unseen_anime_ids, fill_value=0)

    # Combine content-based and collaborative filtering scores
    hybrid_scores = alpha * content_scores + (1 - alpha) * collaborative_scores
    hybrid_scores = hybrid_scores.sort_values(ascending=False).head(items)

    # Get recommendations
    recommended_anime = anime_data[anime_data['anime_id'].isin(hybrid_scores.index)]
    return recommended_anime

# F1 Evaluation Function
def evaluate_recommendations(user_index, recommendations, rating_matrix, ground_truth_threshold=7):
    user_ratings = rating_matrix.loc[user_index]
    ground_truth = set(user_ratings[user_ratings >= ground_truth_threshold].index)
    recommended_items = set(recommendations['anime_id'])
    relevant_recommendations = recommended_items.intersection(ground_truth)

    # Calculate precision, recall, F1
    precision = len(relevant_recommendations) / len(recommended_items) if recommended_items else 0
    recall = len(relevant_recommendations) / len(ground_truth) if ground_truth else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return precision, recall, f1

# Execution
current_user = 51
similar_user_indices = similar_users(current_user, rating_matrix_train)
recommendations = hybrid_recommendation(current_user, similar_user_indices, rating_matrix_train, content_similarity_matrix, anime_data, items=5)
precision, recall, f1 = evaluate_recommendations(current_user, recommendations, rating_matrix_test)

# Output
print(f"Similar Users for User {current_user}: {similar_user_indices}")
similar_users_indices = similar_users(current_user, rating_matrix)
display_top_watched_shows(current_user, similar_users_indices, rating_matrix, anime_data, top_n=5)
print(f"\nRecommendations for User {current_user}:\n{recommendations[['anime_id', 'name', 'rating']]}")
print(f"\nEvaluation Metrics for User {current_user}:")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
