from timm import create_model
import torch.nn as nn
import torchvision.models as models

class resnet(nn.Module):
    def __init__(self,name="resnet50"):
        super(resnet, self).__init__()
        if name=="resnet50":
            model = models.resnet50(pretrained=True)
            self.output_num = 2048
        elif name=="resnet101":
            model = models.resnet101(pretrained=True)
            self.output_num = 2048

        self.model = nn.Sequential(*(list(model.children())[:-1]))
        self.head = model.fc

    def flatten(self,x):
        return nn.Flatten()(x)

    def forward_features(self,x):
        return self.model(x)

    def forward_head(self,x):
        return self.head(self.flatten(x))

    def forward(self,x):
        out = self.forward_features(x)
        return self.forward_head(out)

class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        model = models.alexnet(pretrained=True)
        self.model = nn.Sequential(*(list(model.children())[:-1]))
        self.head = model.classifier
        self.output_num = 9216

    def flatten(self,x):
        return nn.Flatten()(x)

    def forward_features(self,x):
        return self.model(x)

    def forward_head(self,x):
        return self.head(self.flatten(x))

    def forward(self,x):
        out = self.forward_features(x)
        return self.forward_head(out)

class deit(nn.Module):
    def __init__(self,name="deit_base"):
        super(deit, self).__init__()
        assert name=="deit_base" or name=="deit_small" or name=="deit_base_21k"
        if name=="deit_base":
            # model_name = "deit_base_distilled_patch16_224"
            model_name = "vit_base_patch16_224"
            self.output_num = 768
        elif name=="deit_small":
            model_name = "deit_small_distilled_patch16_224"
            self.output_num = 384
        elif name=='deit_base_21k':
            model_name = "deit3_base_patch16_224_in21ft1k"
            self.output_num = 768

        self.model_name = model_name
        self.model = create_model(model_name,pretrained=True)

    def flatten(self,x):
        if self.model_name == "deit3_base_patch16_224_in21ft1k":
            return x[:, 0]
        else:
            # return 0.5 * (x[:, 0] + x[:, 1])
            return x[:, 0]

    def forward_features(self,x):
        return self.model.forward_features(x)

    def forward_head(self,x):
        return self.model.forward_head(x)

    def forward(self, x):
        return self.model(x)

class swint(nn.Module):
    def __init__(self,name="swint_base"):
        super(swint, self).__init__()
        # assert name=="swint_base" or name=="swint_small" or name=="swint_base_v2"or name=="swint_small_v2"
        if name=="swint_base":
            model_name = "swin_base_patch4_window7_224"
            self.output_num = 1024
        elif name=="swint_small":
            model_name = "swin_small_patch4_window7_224"
            self.output_num = 768
        self.model = create_model(model_name,pretrained=True)

    def flatten(self,x):
        return x.mean(1)

    def forward_features(self,x):
        return self.model.forward_features(x)

    def forward_head(self,x):
        return self.model.forward_head(x)

    def forward(self, x):
        return self.model(x)

def get_backbone(name):
    model = None
    if "resnet" in name:
        model = resnet(name)
    elif "deit" in name:
        model = deit(name)
    elif "swint" in name:
        model = swint(name)
    elif "alexnet" in name:
        model = alexnet()
    return model
