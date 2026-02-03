import numpy as np
data = np.load('mlp_weights.npz', allow_pickle=True)
print("Keys:", list(data.keys()))
classes = data['classes']
print("Classes type:", type(classes))
print("Classes content:", classes)
for i, c in enumerate(classes):
    print(f"Index {i}: {c} (type: {type(c)})")
