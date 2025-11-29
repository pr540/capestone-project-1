import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import pickle

# Load data
print("Loading data...")
data = pd.read_csv("./TESS_FEATURES.csv")

# Drop unnecessary column
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Separate features and target
X = data.drop('emotion', axis=1).values
y = data['emotion'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Define pipeline
steps = [('scaler', StandardScaler()),
         ('MLP', MLPClassifier(max_iter=500))] # Increased max_iter to ensure convergence

pipeline_mlp = Pipeline(steps)

# Train model
print("Training model...")
mlp = pipeline_mlp.fit(X_train, y_train)

print('Accuracy: {}'.format(mlp.score(X_test, y_test)))

# Save model
print("Saving model to mlp.pkl...")
with open('mlp.pkl', 'wb') as f:
    pickle.dump(mlp, f)

print("Done.")
