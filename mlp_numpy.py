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
        # Scale and clip to prevent network saturation
        X = (X - self.mean) / (self.scale + 1e-8)
        X = np.clip(X, -10, 10) 
        
        # Forward pass
        curr = X
        for i in range(len(self.w) - 1):
            curr = self.relu(curr @ self.w[i] + self.b[i])
        
        # Final layer
        logits = curr @ self.w[-1] + self.b[-1]
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)[0]
        label = self.classes[idx]
        # Handle numpy string types and ensure full string return
        if isinstance(label, (np.ndarray, np.generic)):
            label = label.item()
        label = str(label)
        # Map internal 'ps' to user-friendly name
        if label == 'ps':
            return 'Pleasant Surprise'
        return label
