import pretrainedmodels
from torch import nn
import torch
from torch.nn.init import kaiming_normal


def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())


def num_features(m):
    c = children(m)
    if len(c) == 0:
        return None
    for l in reversed(c):
        if hasattr(l, 'num_features'):
            return l.num_features
        res = num_features(l)
        if res is not None:
            return res


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        size = size or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def forward(self, input_):
        return input_.view(input_.size(0), -1)


def create_fc_layer(ni, nf, dropout_rate, activations=None):
        res = [nn.BatchNorm1d(num_features=ni)]
        if dropout_rate:
            res.append(nn.Dropout(p=dropout_rate))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if activations:
            res.append(activations)
        return res


def cond_init(m, init_fn):
    if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if hasattr(m, 'weight'):
            init_fn(m.weight)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)


def apply_init(m, init_fn):
    m.apply(lambda x: cond_init(x, init_fn))


def get_resnet34():
    model_name = 'resnet34'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    last_features_count = num_features(model)*2
    layers = list(model.children())[:8]

    w = layers[0].weight.data
    # w = w(torch.device('cuda:0'))
    # print(torch.zeros(64, 1, 7, 7).dtype, w.dtype)
    layers[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    layers[0].weight = torch.nn.Parameter(torch.cat((w, torch.zeros(64, 1, 7, 7)), dim=1))


    layers += [AdaptiveConcatPool2d(), Flatten()]
    # print(layers)
    # print('-' * 50)
    # print(nn.Sequential(*layers))
    # print('-' * 50)

    # model.last_linear = nn.Linear(in_features=2048, out_features=28, bias=True)
    # print(num_features(model))
    # print('-' * 50)
    middle_value = num_features(model)

    head_layers = [*create_fc_layer(last_features_count, middle_value, dropout_rate=0.5, activations=nn.ReLU()),
                   *create_fc_layer(middle_value, 28, dropout_rate=0.5)]
    fc_model = nn.Sequential(*head_layers)
    apply_init(fc_model, kaiming_normal)
    # print('TYPE', type(layers), type(head_layers))

    model = nn.Sequential(*(layers + head_layers))

    return model


def get_resnet101():
    model_name = 'resnet101'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.last_linear = nn.Linear(in_features=8192, out_features=28, bias=True)

    return model


def get_se_resnext101_32x4d():
    model_name = 'se_resnext101_32x4d'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.layer0.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.last_linear = nn.Linear(in_features=2048, out_features=28, bias=True)

    return model


def get_se_resnet50():
    model_name = 'se_resnet50'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.layer0.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.last_linear = nn.Linear(in_features=2048, out_features=28, bias=True)

    return model


def get_resnet34_3_channels():
    model_name = 'resnet34'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.last_linear = nn.Linear(in_features=2048, out_features=28, bias=True)

    return model


def get_resnet34_3_models():
    model0 = get_resnet34_3_channels()
    layers0 = list(model0.children())[:-1]
    layers0 += [AdaptiveConcatPool2d(), Flatten()]

    model1 = get_resnet34_3_channels()
    layers1 = list(model1.children())[:-1]
    layers1 += [AdaptiveConcatPool2d(), Flatten()]

    model2 = get_resnet34_3_channels()
    layers2 = list(model2.children())[:-1]
    layers2 += [AdaptiveConcatPool2d(), Flatten()]

    last_features_count = num_features(model0)*2

    # print(layers0)
    # print('-' * 50)
    # print(nn.Sequential(*layers0))
    # print('-' * 50)

    # model.last_linear = nn.Linear(in_features=2048, out_features=28, bias=True)
    # print(num_features(model))
    # print('-' * 50)
    middle_value = num_features(model0)

    head_layers = [*create_fc_layer(last_features_count, middle_value, dropout_rate=0.5, activations=nn.ReLU()),
                   *create_fc_layer(middle_value, 28, dropout_rate=0.5)]
    fc_model = nn.Sequential(*head_layers)
    apply_init(fc_model, kaiming_normal)
    # print('TYPE', type(layers), type(head_layers))

    model = nn.Sequential(*(layers0 + head_layers))

    return model


class ResNet3(nn.Module):
    def __init__(self):
        super(ResNet3, self).__init__()

        model0 = get_resnet34_3_channels()
        model0.load_state_dict(torch.load('shapovalov/tmp/0-94-0.594-model.pt'))
        self.resnet0 = nn.Sequential(*list(model0.children())[:-1])

        model1 = get_resnet34_3_channels()
        model1.load_state_dict(torch.load('shapovalov/tmp/1-95-0.598-model.pt'))
        self.resnet1 = nn.Sequential(*list(model1.children())[:-1])

        model2 = get_resnet34_3_channels()
        model2.load_state_dict(torch.load('shapovalov/tmp/2-91-0.602-model.pt'))
        self.resnet2 = nn.Sequential(*list(model2.children())[:-1])

        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.mp = nn.AdaptiveMaxPool2d((1,1))

        self.bn = nn.BatchNorm1d(num_features=3*2048)
        self.drop = nn.Dropout(0.5)

        self.fc = nn.Linear(in_features=3*2048, out_features=28, bias=True)

    def forward(self, x, y, z):
        x1 = self.resnet0(x)
        x2 = self.resnet1(y)
        x3 = self.resnet2(z)

        m = torch.cat((x1, x2, x3),1)

        m = m.view(-1,3*2048)

        m = self.bn(m)
        m = self.drop(m)
        m = self.fc(m)

        return m


class ResNet3exp(nn.Module):
    def __init__(self):
        super(ResNet3exp, self).__init__()

        model0 = get_resnet34_3_channels()
        model0.load_state_dict(torch.load('shapovalov/tmp/0-94-0.594-model.pt'))
        self.resnet0 = nn.Sequential(*list(model0.children())[:-1])

        model1 = get_resnet34_3_channels()
        model1.load_state_dict(torch.load('shapovalov/tmp/1-95-0.598-model.pt'))
        self.resnet1 = nn.Sequential(*list(model1.children())[:-1])

        model2 = get_resnet34_3_channels()
        model2.load_state_dict(torch.load('shapovalov/tmp/2-91-0.602-model.pt'))
        self.resnet2 = nn.Sequential(*list(model2.children())[:-1])

        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.mp = nn.AdaptiveMaxPool2d((1,1))

        self.bn = nn.BatchNorm1d(num_features=3*1024)
        self.drop = nn.Dropout(0.5)

        self.fc = nn.Linear(in_features=3*1024, out_features=28, bias=True)

    def forward(self, x, y, z):
        x1 = self.resnet0(x) # 192 * 28 - resnet; 196608 x 2 - resnet[:-1]
        x2 = self.resnet1(y)
        x3 = self.resnet2(z)

        x1 = torch.cat([self.mp(x1), self.ap(x1)], 1)
        x2 = torch.cat([self.mp(x2), self.ap(x2)], 1)
        x3 = torch.cat([self.mp(x3), self.ap(x3)], 1)


        m = torch.cat((x1, x2, x3),1) # 192 * 84 - resnet
        m = self.drop(m)

        m = m.view(-1,3*1024)

        #m = self.bn(m)
        #m = self.drop(m)

        m = self.fc(m)
        return m


class AdaptiveConcatPool2d(nn.Module):
    def init(self, size=None):
        super().init()
        size = size or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


# import os
# os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-9.0/lib64'
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

from torchsummary import summary

if __name__ == '__main__':
    # model_ = get_resnet34()
    # print(model_.eval())
    # print(pretrainedmodels.pretrained_settings['resnet34'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ = ResNet3exp().to(device)


    summary(model_, [(3, 256, 256), (3, 256, 256), (3, 256, 256)])

    #print(model_.eval())
    #x = torch.Tensor(1, 3, 256, 256)  # shape = (batch size, channels, height, width)
    #print(compute_out_size(x.size(), model_))
    #print(pretrainedmodels.pretrained_settings['resnet34'])
