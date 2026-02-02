import numpy as np

class NumpyMLP:
    def __init__(self, weights_path):
        data = np.load(weights_path, allow_pickle=True)
        self.w = data['w']
        self.b = data['b']
        self.mean = data['mean']
        self.scale = data['scale']
        self.classes = data['classes']

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict_proba(self, X):
        # Scale
        X = (X - self.mean) / self.scale
        
        # Forward pass
        curr = X
        for i in range(len(self.w) - 1):
            curr = self.relu(curr @ self.w[i] + self.b[i])
        
        # Final layer
        logits = curr @ self.w[-1] + self.b[-1]
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        # argmax returns an array of indices. We take the first one since we handle 1 sample.
        idx = np.argmax(probs, axis=1)[0]
        return str(self.classes[idx])
