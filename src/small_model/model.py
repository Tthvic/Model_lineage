import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes, pretrained=True, path=None):
    model_name = model_name.lower()
    
    if 'mobilenet' in model_name:
        model = models.mobilenet_v2(pretrained=pretrained)
        # Replace the classifier
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    elif 'resnet' in model_name:
        if '18' in model_name:
            model = models.resnet18(pretrained=pretrained)
        elif '50' in model_name:
            model = models.resnet50(pretrained=pretrained)
        else:
            model = models.resnet18(pretrained=pretrained)
            
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Model {model_name} not supported")

    if path:
        print(f"Loading model weights from {path}")
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)

    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
