import numpy as np
import pickle
from mlp_numpy import NumpyMLP

def test():
    # Load original
    with open('mlp.pkl', 'rb') as f:
        m = pickle.load(f)
    
    # Create fake data
    X = np.random.randn(1, 180)
    
    # Sklearn prediction
    target_probs = m.predict_proba(X)
    target_pred = m.predict(X)
    
    # Numpy prediction
    nm = NumpyMLP('mlp_weights.npz')
    np_probs = nm.predict_proba(X)
    np_pred = nm.predict(X)
    
    print(f"Sklearn Probs: {target_probs[0][:3]}")
    print(f"Numpy Probs:   {np_probs[0][:3]}")
    print(f"Match: {np.allclose(target_probs, np_probs, atol=1e-5)}")
    print(f"Pred match: {target_pred[0] == np_pred}")

test()
