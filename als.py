
Conversation opened. 1 read message.

Skip to content
Using Gmail with screen readers
3 of 60,553
(no subject)
Inbox

manushree tyagi <manushritt@gmail.com>
Sat, May 18, 11:55 PM (6 hours ago)
to me


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from implicit.als import AlternatingLeastSquares
import networkx as nx
from sklearn.model_selection import ParameterGrid
import time
import matplotlib.pyplot as plt

# Read data from CSV
data = pd.read_csv('soc-sign-bitcoinotc.csv', header=None, names=['Source', 'Target', 'Weight', 'Date'])

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

def calculate_coefficients(G):
    common_neighbors = {}
    jaccard_coefficient = {}
    preferential_attachment = {}
    adamic_adar = {}
    resource_allocation = {}
    local_clustering = {}
    
    for idx, edge in enumerate(G.edges()):
        if idx % 1000 == 0:
            print(f"Processed {idx} edges for coefficients calculation.")
        u, v = edge
        
        predecessors_u = set(G.predecessors(u))
        predecessors_v = set(G.predecessors(v))
        
        common_neighbors[edge] = len(predecessors_u & predecessors_v)
        
        union_neighbors = len(predecessors_u | predecessors_v)
        jaccard_coefficient[edge] = common_neighbors[edge] / union_neighbors if union_neighbors != 0 else 0
        
        preferential_attachment[edge] = len(predecessors_u) * len(predecessors_v)
        
        adamic_adar[edge] = sum(1 / np.log(len(list(G.successors(x)))) for x in predecessors_u & predecessors_v if len(list(G.successors(x))) > 1)
        
        resource_allocation[edge] = sum(1 / len(list(G.successors(x))) for x in predecessors_u & predecessors_v if len(list(G.successors(x))) > 0)
        
        local_clustering[u] = nx.clustering(G, u, weight='weight')
        local_clustering[v] = nx.clustering(G, v, weight='weight')
    
    return common_neighbors, jaccard_coefficient, preferential_attachment, adamic_adar, resource_allocation, local_clustering

# Create a directed graph
G = nx.DiGraph()
for row in data.itertuples():
    G.add_edge(row.Source, row.Target, weight=row.Weight)

print("Graph construction completed.")

# Calculate signed weighted clustering coefficient
start_time = time.time()
clustering_coeffs = signed_weighted_clustering_coefficient(G)
average_clustering_coeff = np.mean(list(clustering_coeffs.values()))
print(f'Average Weighted Signed Clustering Coefficient: {average_clustering_coeff}')
print(f'Clustering coefficient calculation time: {time.time() - start_time} seconds')

# Calculate other coefficients
start_time = time.time()
common_neighbors, jaccard_coefficient, preferential_attachment, adamic_adar, resource_allocation, local_clustering = calculate_coefficients(G)
print(f'Coefficient calculation time: {time.time() - start_time} seconds')

# Ground truth: Positive links are those with a positive weight, negative links are those with a negative weight
positive_links = data[data['Weight'] > 0]
negative_links = data[data['Weight'] < 0]

# Prepare the training matrix
num_users = data['Source'].max() + 1
num_items = data['Target'].max() + 1
train_matrix = np.zeros((num_users, num_items))

# Fill in the training matrix
for row in data.itertuples():
    train_matrix[row.Source, row.Target] = row.Weight

# Change the train_matrix to CSR matrix format and transpose it for ALS expects item-user matrix
train_matrix = csr_matrix(train_matrix).T

def predict_sign_als(model, user, item):
    user_vector = model.user_factors[user, :]
    item_vector = model.item_factors[item, :]
    prediction = user_vector.dot(item_vector)
    return 1 if prediction > 0 else -1

def predict_weight_als(model, user, item):
    user_vector = model.user_factors[user, :]
    item_vector = model.item_factors[item, :]
    return user_vector.dot(item_vector)

def generate_weight_predictions(model, data, common_neighbors, jaccard_coefficient, preferential_attachment, adamic_adar, resource_allocation, local_clustering):
    predictions = []
    true_labels = []
    for idx, row in enumerate(data.itertuples()):
        if idx % 1000 == 0:
            print(f"Processed {idx} rows for weight predictions.")
        true_labels.append(int(row.Weight))  
        
        # Extract coefficients for the current edge
        common_neighbors_val = common_neighbors.get((row.Source, row.Target), 0)
        jaccard_coefficient_val = jaccard_coefficient.get((row.Source, row.Target), 0)
        preferential_attachment_val = preferential_attachment.get((row.Source, row.Target), 0)
        adamic_adar_val = adamic_adar.get((row.Source, row.Target), 0)
        resource_allocation_val = resource_allocation.get((row.Source, row.Target), 0)
        local_clustering_source = local_clustering.get(row.Source, 0)
        local_clustering_target = local_clustering.get(row.Target, 0)
        
        # Append features to the feature list
        feature = [
            common_neighbors_val, 
            jaccard_coefficient_val, 
            preferential_attachment_val, 
            adamic_adar_val, 
            resource_allocation_val, 
            local_clustering_source,
            local_clustering_target
            
        ]
        
        # Reshape the feature list to fit the model input
        X_train = np.array(feature).reshape(1, -1)
        
        # Predict the weight using the ALS model
        predicted_weight = predict_weight_als(model, row.Target, row.Source)
        
        # Convert the predicted weight to a binary class label based on a threshold
        threshold = 0  
        predicted_label = 1 if predicted_weight > threshold else -1
        
        predictions.append(predicted_label)
        
    return true_labels, np.array(predictions)

# Grid search for the best parameters
param_grid = {
    'factors': [50],  
    'regularization': [0.01],  
    'iterations': [100]  
}

best_accuracy = 0
best_precision = 0
best_recall = 0
best_params = None

best_mse = np.inf
best_params_mse = None

for params in ParameterGrid(param_grid):
    print(f"Training ALS model with parameters: {params}")
    als = AlternatingLeastSquares(**params)
    als.fit(train_matrix)
    
    true_labels, predictions = generate_weight_predictions(als, data, common_neighbors, jaccard_coefficient, preferential_attachment, adamic_adar, resource_allocation, local_clustering)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    
    mse = mean_squared_error(true_labels, predictions)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_precision = precision
        best_recall = recall
        best_params = params
    
    if mse < best_mse:
        best_mse = mse
        best_params_mse = params

        plt.scatter(true_labels, predictions, color='blue', label='Data Points')  # Scatter plot of data points
plt.plot(true_labels, predictions, color='red', label='Regression Line')  # Plotting the regression line
plt.xlabel('True Labels')  # Label for x-axis
plt.ylabel('Predicted Values')  # Label for y-axis
plt.title('Linear Regression')  # Title of the plot
plt.legend()  # Show legend
plt.grid(True)  # Show grid
plt.show()  # Display the plot

print(f'Best Accuracy: {best_accuracy}')
print(f'Best Precision: {best_precision}')
print(f'Best Recall: {best_recall}')
print(f'Best Parameters for Accuracy: {best_params}')

print(f'Best MSE: {best_mse}')
print(f'Best Parameters for MSE: {best_params_mse}')

print(f'Done!')





