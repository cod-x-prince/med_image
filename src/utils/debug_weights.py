import torch

model_path = 'models/chexnet_model.pth'

try:
    state_dict = torch.load(model_path, map_location='cpu')
    print("✅ Model file loaded successfully!")
    print("Here are some keys in the state_dict:")
    for key in list(state_dict.keys())[:10]:
        print(" -", key)
except Exception as e:
    print("❌ Failed to load the model:")
    print(e)
