import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import networkx as nx
from implicit.als import AlternatingLeastSquares

# Load data
data = pd.read_csv('soc-sign-bitcoinotc.csv', header=None, names=['Source', 'Target', 'Weight', 'Date'])

# Create directed graph
G = nx.DiGraph()
for row in data.itertuples():
    G.add_edge(row.Source, row.Target, weight=row.Weight)

# Calculate clustering coefficient
def signed_weighted_clustering_coefficient(G):
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

# Calculate clustering coefficient
clustering_coeffs = signed_weighted_clustering_coefficient(G)
average_clustering_coeff = np.mean(list(clustering_coeffs.values()))
print(f'Average Weighted Signed Clustering Coefficient: {average_clustering_coeff}')

# Function to compute common neighbors
def common_neighbors(G, u, v):
    return len(set(G.successors(u)).intersection(set(G.predecessors(v))))

# Function to compute Jaccard's Coefficient
def jaccard_coefficient(G, u, v):
    neighbors_u = set(G.successors(u)).union(set(G.predecessors(u)))
    neighbors_v = set(G.successors(v)).union(set(G.predecessors(v)))
    intersection = neighbors_u.intersection(neighbors_v)
    union = neighbors_u.union(neighbors_v)
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

# Function to compute Preferential Attachment
def preferential_attachment(G, u, v):
    return G.degree(u) * G.degree(v)

# Function to compute Adamic-Adar Coefficient
def adamic_adar(G, u, v):
    common_neighbors = set(G.successors(u)).intersection(set(G.predecessors(v)))
    return sum(1 / np.log(G.degree(w)) for w in common_neighbors)

# Function to compute Resource Allocation Index
def resource_allocation(G, u, v):
    common_neighbors = set(G.successors(u)).intersection(set(G.predecessors(v)))
    return sum(1 / G.degree(w) for w in common_neighbors)

# Create the training matrix for ALS
num_users = data['Source'].max() + 1
num_items = data['Target'].max() + 1
train_matrix = np.zeros((num_users, num_items))

# Fill in the training matrix
for row in data.itertuples():
    train_matrix[row.Source, row.Target] = row.Weight

# Convert to sparse matrix format and transpose for ALS
train_matrix = csr_matrix(train_matrix).T

# Train ALS model
als_model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=100)
als_model.fit(train_matrix)

# Function to predict the weight of an edge using ALS
def predict_weight_als(user, item):
    user_vector = als_model.user_factors[user, :]
    item_vector = als_model.item_factors[item, :]
    prediction = user_vector.dot(item_vector)
    return prediction

# Create a DataFrame for features and labels
features = []
labels = []

for row in data.itertuples():
    u, v = row.Source, row.Target
    weight = row.Weight
    
    cn = common_neighbors(G, u, v)
    jc = jaccard_coefficient(G, u, v)
    pa = preferential_attachment(G, u, v)
    aa = adamic_adar(G, u, v)
    ra = resource_allocation(G, u, v)
    cc_u = clustering_coeffs.get(u, 0)
    cc_v = clustering_coeffs.get(v, 0)
    als_prediction = predict_weight_als(v, u)  # ALS expects item-user matrix

    features.append([cn, jc, pa, aa, ra, cc_u, cc_v, als_prediction])
    labels.append(weight)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict on test set
y_pred = reg.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f'Combined Model Mean Squared Error: {mse}')

# Verify the nodes in the graph
print("Nodes in the graph:", list(G.nodes)[:10])  # Print first 10 nodes for verification

# Example of making a prediction for a single edge using valid node IDs
valid_node_u = list(G.nodes)[0]  # First node
valid_node_v = list(G.nodes)[1]  # Second node

test_features = [[
    common_neighbors(G, valid_node_u, valid_node_v),
    jaccard_coefficient(G, valid_node_u, valid_node_v),
    preferential_attachment(G, valid_node_u, valid_node_v),
    adamic_adar(G, valid_node_u, valid_node_v),
    resource_allocation(G, valid_node_u, valid_node_v),
    clustering_coeffs.get(valid_node_u, 0),
    clustering_coeffs.get(valid_node_v, 0),
    predict_weight_als(valid_node_v, valid_node_u)
]]

predicted_weight = reg.predict(test_features)
print(f'Predicted weight for edge ({valid_node_u}, {valid_node_v}): {predicted_weight}')

