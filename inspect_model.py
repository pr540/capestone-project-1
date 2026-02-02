import pickle
import numpy as np

def inspect_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Check if it's a Pipeline
        if hasattr(model, 'steps'):
            print("Pipeline steps:", [s[0] for s in model.steps])
            # Usually [('scaler', StandardScaler()), ('MLP', MLPClassifier())]
            scaler = model.named_steps.get('scaler')
            mlp = model.named_steps.get('MLP')
        else:
            scaler = None
            mlp = model
            
        if mlp:
            print("MLP Layers:", [c.shape for c in mlp.coefs_])
            print("MLP Activation:", mlp.activation)
            
        if scaler:
            print("Scaler mean shape:", scaler.mean_.shape)
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

inspect_model('mlp.pkl')
