from torchvision import models
import torch.nn as nn

def resnet50(option, class_names, device='cpu'):
    if option not in ['og', 'transfer', 'finetune']:
        raise ValueError(option, 'must be one of [\'og\', \'transfer\', \'finetune\']')

    if option == 'og':
        model_ft = models.resnet50(pretrained=False, progress=True)
    elif option == 'transfer':
        model_ft = models.resnet50(pretrained=True, progress=True)
        # fix model weights
        for param in model_ft.parameters():
            param.requires_grad = False
    elif option == 'finetune':
        model_ft = models.resnet50(pretrained=True, progress=True)

    # reset final fully connected layer
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    return model_ft