# Original resnet method from Spijkervet implementation.
import torchvision


def get_resnet_pt(name, pretrained=False):
    """Retrieve Resnet from Pytorch Library"""
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]