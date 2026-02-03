import numpy as np
data = np.load('mlp_weights.npz', allow_pickle=True)
print("Mean (first 5):", data['mean'][:5])
print("Scale (first 5):", data['scale'][:5])
