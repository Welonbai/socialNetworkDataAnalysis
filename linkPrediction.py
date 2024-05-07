import pandas as pd
from sklearn.model_selection import train_test_split
import networkx as nx
from math import log
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Read data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Create graph
G = nx.Graph()
nodes = set(train_data['node1']).union(set(train_data['node2'])).union(set(test_data['node1'])).union(set(test_data['node2']))  # Get all nodes
G.add_nodes_from(nodes)

edges = [(row['node1'], row['node2']) for index, row in train_data.iterrows() if row['label'] == 1]
G.add_edges_from(edges)

def calculate_common_neighbor(row, graph):
    node1 = row['node1']
    node2 = row['node2']
    if node1 not in graph or node2 not in graph:
        return 0
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    return len(neighbors1.intersection(neighbors2))

def calculate_jaccard_coefficient(row, graph):
    node1 = row['node1']
    node2 = row['node2']
    if node1 not in graph or node2 not in graph:
        return 0
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    if not neighbors1 or not neighbors2:
        return 0
    intersection = len(neighbors1.intersection(neighbors2))
    union = len(neighbors1.union(neighbors2))
    return intersection / union

def calculate_shortest_path_length(row, graph):
    node1 = row['node1']
    node2 = row['node2']
    if node1 not in graph or node2 not in graph:
        return -1  # Return -1 if nodes are not in the graph
    try:
        shortest_path_length = nx.shortest_path_length(graph, source=node1, target=node2)
    except nx.NetworkXNoPath:
        shortest_path_length = -1  # Return -1 if there is no path between the two nodes
    return shortest_path_length

def calculate_adamic_adar(row, graph):
    node1 = row['node1']
    node2 = row['node2']
    if node1 not in graph or node2 not in graph:
        return 0
    common_neighbors = set(graph.neighbors(node1)).intersection(set(graph.neighbors(node2)))
    adamic_adar = sum(1 / (log(len(set(graph.neighbors(neighbor))))) for neighbor in common_neighbors)
    return adamic_adar

def generate_features(data):
    # Calculate common neighbor
    data['common_neighbor'] = data.apply(lambda row: calculate_common_neighbor(row, G), axis=1)
    
    # Calculate Jaccard's coefficient
    data['jaccard_coefficient'] = data.apply(lambda row: calculate_jaccard_coefficient(row, G), axis=1)
    
    # Calculate shortest path
    data['shortest_path_length'] = data.apply(lambda row: calculate_shortest_path_length(row, G), axis=1)
    
    # Calculate adamic_adar
    data['adamic_adar'] = data.apply(lambda row: calculate_adamic_adar(row, G), axis=1)

    return data

train_data = generate_features(train_data)

X = train_data[['common_neighbor', 'jaccard_coefficient', 'shortest_path_length', 'adamic_adar', 'node1', 'node2']]
y = train_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest model
rf_model = RandomForestClassifier()

# Create SVM model
svm_model = SVC(probability=True)

# Create Logistic Regression model
lr_model = LogisticRegression()

# Create Voting Classifier ensemble model
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model), ('lr', lr_model)], voting='soft')

# Fit the model
ensemble_model.fit(X_train, y_train)

# Make predictions
predictions = ensemble_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Generate features for the test set
test_data = generate_features(test_data)

# Make predictions
XComplete = test_data[['common_neighbor', 'jaccard_coefficient', 'shortest_path_length', 'adamic_adar', 'node1', 'node2']]
realPredictions = ensemble_model.predict(XComplete)

# Output results to csv
sample_submission = pd.DataFrame({'idx': test_data['idx'], 'ans': realPredictions})
sample_submission.to_csv('sample_submission.csv', index=False)

