import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from implicit.als import AlternatingLeastSquares
import networkx as nx


def signed_weighted_clustering_coefficient(G):
  """
  Calculates signed weighted clustering coefficient for each node.
  """
  clustering_coeffs = {}
  for node in G.nodes():
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
      clustering_coeffs[node] = 0.0
      continue

    triangles = 0
    total_triplets = 0

    for i in range(len(neighbors)):
      for j in range(i + 1, len(neighbors)):
        u, v = neighbors[i], neighbors[j]
        if G.has_edge(u, v):
          total_triplets += 1
          if (G.has_edge(u, node) and G.has_edge(node, v) and G.has_edge(u, v)) and (G[u][node]['weight'] * G[node][v]['weight'] * G[u][v]['weight']) > 0:
            triangles += 1

    if total_triplets == 0:
      clustering_coeffs[node] = 0.0
    else:
      clustering_coeffs[node] = triangles / total_triplets

  return clustering_coeffs


def signed_weighted_cn(G, source, target):
  """
  Calculates signed Common Neighbors (CN) for an edge.
  """
  # Find common neighbors with positive weights
  common_neighbors = list(set(G.predecessors(source)) & set(G.successors(target)))
  cn_weight = 0
  for neighbor in common_neighbors:
    cn_weight += G[source][neighbor]['weight'] * G[neighbor][target]['weight']
  return cn_weight


def jaccard_coefficient(G, source, target):
  """
  Calculates Jaccard's Coefficient for an edge.
  """
  # Find all neighbors (positive and negative weights)
  source_neighbors = list(G.predecessors(source)) + list(G.successors(source))
  target_neighbors = list(G.predecessors(target)) + list(G.successors(target))
  # Count intersection and union considering weight signs
  intersection_weight = 0
  union_weight = 0
  for neighbor in set(source_neighbors) & set(target_neighbors):
    if G.has_edge(source, neighbor):
      intersection_weight += G[source][neighbor]['weight']
    if G.has_edge(target, neighbor):
      intersection_weight += G[target][neighbor]['weight']
  for neighbor in set(source_neighbors) | set(target_neighbors):
    if G.has_edge(source, neighbor):
      union_weight += abs(G[source][neighbor]['weight'])
    if G.has_edge(target, neighbor):
      union_weight += abs(G[target][neighbor]['weight'])
  if union_weight == 0:
    return 0
  return intersection_weight / union_weight


# Data loading
data = pd.read_csv('soc-sign-bitcoinotc.csv', header=None, names=['Source', 'Target', 'Weight', 'Date'])


# Create directed graph
G = nx.DiGraph()
for row in data.itertuples():
  G.add_edge(row.Source, row.Target, weight=row.Weight)


# Calculate signed weighted clustering coefficient
clustering_coeffs = signed_weighted_clustering_coefficient(G)
average_clustering_coeff = np.mean(list(clustering_coeffs.values()))

print(f'Average Weighted Signed Clustering Coefficient: {average_clustering_coeff}')


# Ground truth: positive links are those with positive weight, negative links are those with negative weight
positive_links = data[data['Weight'] > 0]
negative_links = data[data['Weight'] < 0]


# Create training matrix with additional features (CN and Jaccard)
num_users = data['Source'].max() + 1
num_items = data['Target'].max() + 1
train_matrix = np.zeros((num_users, num_items))

# Fill training matrix and calculate CN and Jaccard for each edge
for row in data.itertuples():
  source, target, weight = row.Source, row.Target, row.Weight
  cn_value = signed_weighted_cn(G, source, target)
  jaccard_value = jaccard_coefficient(G, source, target)
  train_matrix[source, target] = weight  # Include weight
  # Add CN and Jaccard as additional features
  train_matrix[source, target] = [weight, cn_value, jaccard_value]

# Convert to csr_matrix and transpose for ALS
train_matrix = csr_matrix(train_matrix).T


# Hyperparameter search for ALS (same as before)
from sklearn.model_selection import ParameterGrid

param_grid = {
  'factors': [50],  # Adjust factors as needed
  'regularization': [0.01],  # Adjust regularization as needed
  'iterations': [100],  # Adjust iterations as needed
}


def predict_sign_als(model, user, item):
  user_vector = model.user_factors[user, :]
  item_vector = model.item_factors[item, :]
  prediction = user_vector.dot(item_vector)
  return 1 if prediction > 0 else -1


def generate_predictions(model, data):
  predictions = []
  true_labels = []
  for row in data.itertuples():
    true_labels.append(1 if row.Weight > 0 else -1)
    predicted_sign = predict_sign_als(model, row.Target, row.Source)
    predictions.append(predicted_sign)
  return true_labels, predictions


# Grid search for the best parameters (same as before)
best_accuracy = 0
best_precision = 0
best_recall = 0
best_params = None

for params in ParameterGrid(param_grid):
  als = AlternatingLeastSquares(**params)
  als.fit(train_matrix)
  true_labels, predictions = generate_predictions(als, data)
  accuracy = accuracy_score(true_labels, predictions)
  precision = precision_score(true_labels, predictions, average='macro')
  recall = recall_score(true_labels, predictions, average='macro')
  if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_precision = precision
    best_recall = recall
    best_params = params

print(f'Best Accuracy: {best_accuracy}')
print(f'Best Precision: {best_precision}')
print(f'Best Recall: {best_recall}')
print(f'Best Parameters: {best_params}')
print(f'Best Model: {als}')
# print(f'Best Model User Factors: {als.user_factors}')
# print(f'Best Model Item Factors: {als.item_factors}')
# print(f'Best Model User Factors Shape: {als.user_factors.shape}')
# print(f'Best Model Item Factors Shape: {als.item_factors.shape}')
print(f'Done!')
