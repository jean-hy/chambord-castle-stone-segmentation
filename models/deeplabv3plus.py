import torch
import torch.nn as nn
import torchvision

def get_deeplabv3_plus(num_classes=2, pretrained=True):
    """
    Returns a DeepLabv3 model with a ResNet-50 backbone.
    Adjust the final classifier to have `num_classes` outputs.
    """
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    # Replace the classifier with a new one
    model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
    return model
