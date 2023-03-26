from torchvision.models.resnet import resnet18
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, model_path, pretrained=True, num_classes=7):
        super(Model, self).__init__()
        resnet = resnet18(pretrained=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        weights = {key: arr for key, arr in checkpoint['state_dict'].items()}
        resnet.load_state_dict(weights)

        self.features = nn.Sequential(*list(resnet.children())[:-2]) # just the backbone
        self.features2 = nn.Sequential(*list(resnet.children())[-2:-1])  # average pooling
        self.fc = nn.Linear(512, 7)

    def forward(self, x):
        x = self.features(x)
        #### bs, 512, 4, 4
        feature = self.features2(x)
        #### bs, 512, 1, 1

        feature = feature.view(feature.size(0), -1)  # flatten
        output = self.fc(feature)

        return output