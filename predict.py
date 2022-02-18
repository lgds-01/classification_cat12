from utils import predict


test_path='cat_12/cat_12_test'
result_path='result.csv'

# MLP
# from models import MLP
# model_path='weights/mlp.pth'

# CNN
# from models import CNN
# model_path='weights/cnn.pth'

# VGG
# from models import VGG
# model_path='weights/vgg.pth'

# pretrained VGG
# from pre_models import VGG
# model_path='weights/vgg_pre.pth'

# ResNet
# from models import ResNet
# model_path='weights/resnet.pth'

# pretrained ResNet
from pre_models import ResNet
model_path='weights/resnet_pre.pth'

# ViT
# from models import ViT
# model_path='weights/vit.pth'


predict(model_path,test_path,result_path)