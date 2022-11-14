from model.architectures.VGG16_model import create_model_VGG16
from model.architectures.VGG16BW_model import create_model_VGG16BW
from model.architectures.AlexNet_model import create_model_AlexNet
from model.architectures.AlexNetBW_model import create_model_AlexNetBW
from model.architectures.ResNet50_model import create_model_ResNet50
from model.architectures.ResNet50BW_model import create_model_ResNet50BW
from model.architectures.EfficientNetB0_model import create_model_EfficientNetB0
from model.architectures.EfficientNetB0BW_model import create_model_EfficientNetB0BW
from model.architectures.EfficientNetB4_model import create_model_EfficientNetB4
from model.architectures.EfficientNetB4BW_model import create_model_EfficientNetB4BW
from model.architectures.EfficientNetB7_model import create_model_EfficientNetB7
from model.architectures.EfficientNetB7BW_model import create_model_EfficientNetB7BW

MODELS: dict = {
    "VGG16": create_model_VGG16,
    "VGG16BW": create_model_VGG16BW,
    "AlexNet": create_model_AlexNet,
    "AlexNetBW": create_model_AlexNetBW,
    "ResNet50": create_model_ResNet50,
    "ResNet50BW": create_model_ResNet50BW,
    "EfficientNetB0": create_model_EfficientNetB0,
    "EfficientNetB0BW": create_model_EfficientNetB0BW,
    "EfficientNetB4": create_model_EfficientNetB4,
    "EfficientNetB4BW": create_model_EfficientNetB4BW,
    "EfficientNetB7": create_model_EfficientNetB7,
    "EffcientNetB7BW": create_model_EfficientNetB7BW,
}
