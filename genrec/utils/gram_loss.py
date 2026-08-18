"""VGG19 Gram-matrix style loss.

Ported from Difix3D/src/loss.py (which is not importable as a Python package,
no __init__.py). Expects VGG19.features (torchvision) as the feature extractor;
layer indices 3/8/15/22/29 correspond to relu1_2..relu5_3.
"""
import torch

STYLE_WEIGHTS = {
    'relu1_2': 1.0 / 2.6,
    'relu2_2': 1.0 / 4.8,
    'relu3_3': 1.0 / 3.7,
    'relu4_3': 1.0 / 5.6,
    'relu5_3': 10.0 / 1.5,
}

_LAYER_NAMES = {
    '3': 'relu1_2',
    '8': 'relu2_2',
    '15': 'relu3_3',
    '22': 'relu4_3',
    '29': 'relu5_3',
}


def get_features(image, model, layers=None):
    if layers is None:
        layers = _LAYER_NAMES
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    b, d, h, w = tensor.size()
    t = tensor.view(b * d, h * w)
    return torch.mm(t, t.t())


def gram_loss(style, target, model):
    style_features = get_features(style, model)
    target_features = get_features(target, model)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    total_loss = 0
    for layer, weight in STYLE_WEIGHTS.items():
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        _, d, h, w = target_feature.shape
        layer_loss = weight * torch.mean((target_gram - style_gram) ** 2)
        total_loss = total_loss + layer_loss / (d * h * w)
    return total_loss
