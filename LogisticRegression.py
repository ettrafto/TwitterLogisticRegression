import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Twitter Edge List
file_path = "./twitter/twitter_combined.txt"  # Adjust if needed
df_edges = pd.read_csv(file_path, sep=' ', header=None, names=['source', 'target'])

# Print basic statistics about the dataset
print(f"Number of edges: {len(df_edges)}")
print(f"Number of unique nodes: {len(pd.unique(df_edges[['source', 'target']].values.ravel()))}")

# Step 2: Generate Node Features
# Aggregate in-degree and out-degree counts
in_degree = df_edges.groupby('target').size().reset_index(name='in_degree')
out_degree = df_edges.groupby('source').size().reset_index(name='out_degree')

# Combine in-degree and out-degree into a single dataframe
df_nodes = pd.DataFrame(pd.unique(df_edges[['source', 'target']].values.ravel()), columns=['node'])
df_nodes = pd.merge(df_nodes, in_degree, left_on='node', right_on='target', how='left').drop(columns=['target'])
df_nodes = pd.merge(df_nodes, out_degree, left_on='node', right_on='source', how='left').drop(columns=['source'])

# Fill NaN values with 0 for nodes without edges
df_nodes['in_degree'] = df_nodes['in_degree'].fillna(0)
df_nodes['out_degree'] = df_nodes['out_degree'].fillna(0)

# Generate synthetic labels for nodes (e.g., "spam" or "not spam")
df_nodes['label'] = np.random.choice(['spam', 'not spam'], size=len(df_nodes))
print(df_nodes.head())

# Step 3: Encode Text Features using TF-IDF (Simulating Posts with Node IDs)
vectorizer = TfidfVectorizer(max_features=500)
df_nodes['post'] = "Node " + df_nodes['node'].astype(str) + " with in-degree " + df_nodes['in_degree'].astype(str)
X = vectorizer.fit_transform(df_nodes['post']).toarray()
y = np.array([1 if label == 'spam' else 0 for label in df_nodes['label']])

# Step 4: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

# Gradient Clipping Function
def L2_clip(v, clip_norm):
    norm = np.linalg.norm(v, ord=2)
    return v if norm <= clip_norm else clip_norm * (v / norm)

# Logistic Loss and Gradient
def loss(theta, xi, yi):
    exponent = -yi * (xi.dot(theta))
    return np.log(1 + np.exp(exponent))

def gradient(theta, xi, yi):
    exponent = yi * (xi.dot(theta))
    return -yi * xi / (1 + np.exp(exponent))

# Noisy Gradient Descent with Differential Privacy
def noisy_gradient_descent(X, y, iterations, epsilon, delta, learning_rate=0.01, clip_norm=2.0, batch_size=64):
    theta = np.zeros(X.shape[1])  # Initialize parameters to zero
    sensitivity = clip_norm / batch_size  # Scaled sensitivity for batches
    n_batches = len(y) // batch_size

    for it in range(iterations):
        perm = np.random.permutation(len(y))  # Shuffle data
        X_shuffled, y_shuffled = X[perm], y[perm]

        for batch_start in range(0, len(y), batch_size):
            batch_end = min(batch_start + batch_size, len(y))
            X_batch = X_shuffled[batch_start:batch_end]
            y_batch = y_shuffled[batch_start:batch_end]
            
            # Compute gradients for the batch
            gradients = np.array([L2_clip(gradient(theta, X_batch[i], y_batch[i]), clip_norm) for i in range(len(y_batch))])
            grad_sum = np.sum(gradients, axis=0)
            
            # Add Gaussian noise to the aggregated gradient
            noise = np.random.normal(0, sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon, size=grad_sum.shape)
            noisy_grad = grad_sum + noise
            
            # Update parameters
            theta -= learning_rate * noisy_grad / len(y_batch)
        
        # Optional: Compute and print loss for the epoch
        current_loss = np.mean([loss(theta, X[i], y[i]) for i in range(len(y))])
        print(f"Iteration {it + 1}/{iterations}, Loss: {current_loss:.4f}")
    
    return theta

# Hyperparameters
iterations = 50
epsilon = 2.0  # Increased privacy budget
delta = 1e-5
clip_norm = 2.0  # Increased clipping norm
learning_rate = 0.5
batch_size = 64

# Train Model with Differential Privacy
theta_dp = noisy_gradient_descent(X_train, y_train, iterations, epsilon, delta, learning_rate, clip_norm, batch_size)

# Evaluate Model on Test Data
def predict(xi, theta):
    return 1 if xi.dot(theta) >= 0 else 0

y_pred = [predict(X_test[i], theta_dp) for i in range(len(y_test))]
accuracy = np.mean(np.array(y_pred) == y_test)
print(f"Test Accuracy with Differential Privacy: {accuracy:.4f}")