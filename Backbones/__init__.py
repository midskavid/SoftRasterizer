from Backbones.resnet import resnet50


def get_backbone(path):
    nn_encoder = resnet50(path)
    nn_decoder = None
    return nn_encoder, nn_decoder
