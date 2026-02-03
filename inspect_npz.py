import numpy as np

def inspect_npz(path):
    try:
        data = np.load(path, allow_pickle=True)
        print("Keys in NPZ:", list(data.keys()))
        if 'classes' in data:
            print("Classes in model:", data['classes'])
            print("Number of classes:", len(data['classes']))
        if 'w' in data:
            print("Weights (W) count:", len(data['w']))
            for i, w in enumerate(data['w']):
                print(f"  Layer {i} W shape: {w.shape}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

inspect_npz('mlp_weights.npz')
