import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1).values  # Convert to numpy array
y = df["Class"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Manual standardization (like StandardScaler)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# Add bias term (intercept column of ones)
X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Cost function (binary cross-entropy)
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(X @ weights)
    epsilon = 1e-5  # Prevent log(0)
    cost = (-1 / m) * np.sum(
        y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
    )
    return cost


# Gradient descent
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        # Predictions
        h = sigmoid(X @ weights)

        # Gradient
        gradient = (1 / m) * (X.T @ (h - y))

        # Update weights
        weights = weights - learning_rate * gradient

        # Track cost
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return weights, cost_history


# Initialize weights (random small values)
np.random.seed(42)
weights = np.random.randn(X_train_scaled.shape[1]) * 0.01

# Train model
learning_rate = 0.01
iterations = 1000
weights, cost_history = gradient_descent(
    X_train_scaled, y_train, weights, learning_rate, iterations
)


# Prediction function
def predict(X, weights, threshold=0.25):
    probabilities = sigmoid(X @ weights)
    return (probabilities >= threshold).astype(int), probabilities


# Evaluate on test set
y_pred, y_proba = predict(X_test_scaled, weights)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model components
model_params = {"weights": weights, "mean": mean, "std": std}
joblib.dump(model_params, "fraud_model_manual.joblib")

print("\n✅ Model trained and saved successfully!")
print(f"Final cost: {cost_history[-1]:.4f}")
print(f"Model weights shape: {weights.shape}")

metrics = {
    "model": "Logistic Regression (manual)",
    "threshold": 0.25,
    "accuracy": float((y_pred == y_test).mean()),
    "precision": float(
        classification_report(y_test, y_pred, output_dict=True)["1"]["precision"]
    ),
    "recall": float(
        classification_report(y_test, y_pred, output_dict=True)["1"]["recall"]
    ),
    "f1": float(
        classification_report(y_test, y_pred, output_dict=True)["1"]["f1-score"]
    ),
}

joblib.dump(metrics, "metrics.joblib")
