import numpy as np
data = np.load('mlp_weights.npz', allow_pickle=True)
classes = data['classes']
print("ORDER:" + ",".join([str(c) for c in classes]))
