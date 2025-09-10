#########################################################################################
#                           AI Assignment                                               #
# Authers : Tamara Ghatashe     2035311                                                 #
#           Rama Ghyadaa        2036789                                                 #
#           Leen Qasem          2034562                                                 # 
#########################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# Fix seed for reproducibility
np.random.seed(20)

# Read CSV file into a DataFrame
data = pd.read_csv("animals.csv")

def sentiment_label(text_data):
    """
    Encodes sentiment labels based on keywords in the text.

    Args:
        text_data (str): The text to analyze.

    Returns:
        int: The encoded sentiment label (1 for Solitary/Herbivore, 2 for Semi-social/Omnivore, 3 for others).
    """
    if "Social" in text_data or "Herbivore" in text_data:
        return 1
    elif "Semi-social" in text_data or "Omnivore" in text_data:
        return 2
    else:
        return 3
  
# Extract numerical features ( columns 2 to 4 contain numerical data)
numerical_features = data.iloc[:, 2:5].values

# Normalize numerical features using MinMaxScaler
scaler = MinMaxScaler()
numerical_features_normalized = scaler.fit_transform(numerical_features)

# Create lists to store animal data
Habit = []
Diet = []
socialBehavior = []
Names = []

# Access animal names and other data
for index, row in data.iterrows():
    Names.append(row.iloc[1]) #Names
    Habit.append(row.iloc[-1]) #Habits
    Diet.append(row.iloc[6])   #Diet
    socialBehavior.append(row.iloc[5]) #socialBehavior

# Categorical data
# One Hot Encoding
data_series = pd.Series(Habit)
encodedHabit = pd.get_dummies(data_series)
encodedHabitInt = encodedHabit.astype(int)

le_habit = np.argmax(encodedHabitInt, axis=1)
scaler = MinMaxScaler()
nurmlizeHabit = scaler.fit_transform(np.array(le_habit).reshape(-1, 1)).flatten()

    
# Label Encoding

encodedDitelabels = [sentiment_label(text) for text in Diet]
encodedsocialBehaviorlabels = [sentiment_label(text) for text in socialBehavior]

nurmlizeDiet = scaler.fit_transform(np.array(encodedDitelabels).reshape(-1, 1)).flatten()

nurmlizeSocialBehaviorlabels = scaler.fit_transform(np.array(encodedsocialBehaviorlabels).reshape(-1, 1)).flatten()

# Combine features
combined_array = np.column_stack([numerical_features_normalized, nurmlizeHabit, nurmlizeDiet, nurmlizeSocialBehaviorlabels])
combined_array_withnames=np.column_stack([Names, combined_array])
#########################################################
# PCA (exclude names)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(combined_array)

# Combine principal components with names (for printing cluster membership)
principalComponents_withnames = np.column_stack([Names, principalComponents])
#########################################################
# K-means
def k_means(data, k):
  """
  K-Means clustering .

  Args:
      data (numpy.ndarray): The data to be clustered.
      k (int): The number of clusters.

  Returns:
      tuple: A tuple containing the centroids and labels.
  """

  # Initialize centroids randomly
  centroids = data[np.random.choice(data.shape[0], k, replace=False)]

  # Initialize labels with all -1 (un assigned)
  labels = np.full(data.shape[0], -1)

  # Main K-Means loop: Continue iterating until centroids don't change significantly
  old_centroids = None
  while not np.array_equal(old_centroids, centroids):
    old_centroids = centroids.copy()

    # Assign data points to the closest centroid
    for i in range(data.shape[0]):
      distances = np.linalg.norm(data[i] - centroids, axis=1)
      labels[i] = np.argmin(distances)

    # Recompute centroids based on assigned data points
    for j in range(k):
      data_points_in_cluster = data[labels == j]
      if data_points_in_cluster.any():  # Check if there are any points in the cluster
        centroids[j] = np.mean(data_points_in_cluster, axis=0)

  return centroids, labels

# K-Means with k=6 
k = 5
centroids, labels = k_means(principalComponents, k)

# Plot clusters
plt.figure(figsize=(8, 6))
for i in range(k):
  # Select data points belonging to cluster i
  cluster_data = principalComponents[labels == i]
  plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, label=f'Cluster {i+1}')


# Plot clusters (continued)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=150, marker='x', label='Centroids')  # Plot centroids
plt.title('Clusters of Animals (Custom K-Means, k={})'.format(k))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show(
