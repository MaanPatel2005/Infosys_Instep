import sys
import os
import torch

# Add the module path dynamically
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/Depth-Anything-V2'))
if module_path not in sys.path:
    sys.path.append(module_path)

from depth_anything_v2.dpt import DepthAnythingV2

# Vérifie si mps est disponible
if torch.backends.__dict__.get('mps', None) and torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    # Other configurations are not needed since we are only using 'vits'
}

def load_model():
    encoder = 'vits'  # Utilise toujours 'vits'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'models/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model

# Test loading the model
if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully")
