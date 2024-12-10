# Twitter Network Analysis with Differentially Private Logistic Regression

## Project Overview
This project explores the application of differential privacy in training logistic regression models on a Twitter network dataset. It focuses on preserving privacy while analyzing and classifying nodes in a directed graph. The workflow involves loading a Twitter edge list, generating node features, encoding features with TF-IDF, and implementing a custom noisy gradient descent algorithm for differentially private learning.

---

## Features
- **Graph Analysis**: Analyze and extract in-degree and out-degree features from a Twitter edge list.
- **Synthetic Data**: Simulate spam detection with synthetic labels and text data for nodes.
- **Differential Privacy**: Train a logistic regression model using a noisy gradient descent algorithm that guarantees differential privacy.
- **Node Classification**: Predict synthetic labels ('spam' or 'not spam') with the trained model.
- **Customizable Privacy Parameters**: Adjust the privacy budget (`epsilon`), sensitivity (`clip_norm`), and other parameters to control the tradeoff between privacy and utility.

---

## Key Components
### 1. **Dataset**
The dataset is a directed edge list from a Twitter network:
- Each line represents a directed edge (`source`, `target`) between two Twitter users.
- Example:  



### 2. **Node Feature Engineering**
- Calculated **in-degree** and **out-degree** for each node.
- Created synthetic labels (`spam` or `not spam`) for classification.

### 3. **Feature Encoding**
- Encoded text data (based on node IDs and degrees) into numerical vectors using TF-IDF (`TfidfVectorizer`).

### 4. **Differentially Private Logistic Regression**
- Implemented a custom **Noisy Gradient Descent** algorithm:
- **Gradient Clipping**: L2-norm clipping to bound individual contributions.
- **Gaussian Noise**: Added noise to gradients to ensure differential privacy.
- Parameters:
  - `epsilon`: Privacy budget controlling the amount of noise.
  - `delta`: Probability of privacy breach.
  - `clip_norm`: Maximum gradient norm for clipping.
- Trained the model over multiple iterations with mini-batches.

### 5. **Evaluation**
- Measured test accuracy and used synthetic data for performance metrics.
- Compared predictions with ground truth labels to assess classification accuracy.

---

## Dependencies
- Python 3.8 or higher
- Libraries:
- `pandas`
- `numpy`
- `scikit-learn`

Install dependencies via pip:
```bash
pip install pandas numpy scikit-learn
```
## Usage Instructions
1. Load Dataset
Ensure the edge list file is in the specified path (./twitter/twitter_combined.txt).

2. Run the Script
Execute the script to:

Preprocess the dataset.
Train the logistic regression model with differential privacy.
Evaluate model performance.
3. Adjust Hyperparameters
Modify key parameters in the script to experiment with privacy and utility tradeoffs:

## Differential Privacy Parameters:
epsilon: Increase for less noise (reduced privacy risk).
delta: Decrease for stricter privacy.
clip_norm: Adjust to control gradient sensitivity.
Training Parameters:
iterations: Number of training epochs.
learning_rate: Step size for gradient updates.
batch_size: Number of samples per batch.
4. Evaluate Results
Check console output for:

### Training loss per iteration.
Test accuracy with differentially private logistic regression.
Example Output
Sample output during training:

```yaml
Copy code
Number of edges: 10000
Number of unique nodes: 5000
Training Data Shape: (4000, 500)
Test Data Shape: (1000, 500)
Iteration 1/50, Loss: 0.6931
Iteration 2/50, Loss: 0.6748
...
Iteration 50/50, Loss: 0.5213
Test Accuracy with Differential Privacy: 0.8670
```