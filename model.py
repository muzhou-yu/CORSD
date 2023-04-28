import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class RN1(nn.Module):
    def __init__(self, C, num_classes):
        super(RN1,self).__init__()

        self.channel_2 = 2 * C
        self.channel_4 = 4 * C
        self.channel_8 = 8 * C
        self.channel_16 = 16 * C

        self.attention1 = nn.Sequential(
            SepConv(
                channel_in= self.channel_2,
                channel_out= self.channel_2
            ),
            nn.BatchNorm2d(self.channel_2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.scala1 = nn.Sequential(
                SepConv(
                    channel_in = self.channel_2,
                    channel_out = self.channel_4
                ),
                SepConv(
                    channel_in = self.channel_4,
                    channel_out = self.channel_8
                ),
                SepConv(
                    channel_in = self.channel_8,
                    channel_out = self.channel_16
                ),
                nn.AdaptiveAvgPool2d(1)
        )

        self.fcout = nn.Sequential(
                nn.Linear(self.channel_16,self.channel_2),
                nn.BatchNorm1d(self.channel_2),
                nn.ReLU(),
                nn.Linear(self.channel_2,1)
                )

        self.auxiliary1 = nn.Linear(self.channel_16, num_classes)

    def forward(self,x):
        x = self.scala1(x)
        combine_feature = x.view(x.size(0),-1)
        x1 = self.fcout(combine_feature)
        x2 = self.auxiliary1(combine_feature)
        return nn.Sigmoid()(x1), x2


class RN2(nn.Module):
    def __init__(self, C, num_classes):
        super(RN2,self).__init__()

        self.channel_2 = 2 * C
        self.channel_4 = 4 * C
        self.channel_8 = 8 * C
        self.channel_16 = 16 * C

        self.attention2 = nn.Sequential(
            SepConv(
                channel_in= self.channel_4,
                channel_out= self.channel_4
            ),
            nn.BatchNorm2d(self.channel_4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.scala2 = nn.Sequential(
                SepConv(
                    channel_in = self.channel_4,
                    channel_out = self.channel_8
                ),
                SepConv(
                    channel_in = self.channel_8,
                    channel_out = self.channel_16
                ),
                nn.AdaptiveAvgPool2d(1)
        )

        self.fcout = nn.Sequential(
                nn.Linear(self.channel_16, self.channel_2),
                nn.BatchNorm1d(self.channel_2),
                nn.ReLU(),
                nn.Linear(self.channel_2,1)
                )

        self.auxiliary2 = nn.Linear(self.channel_16, num_classes)

    def forward(self,x):
        x = self.scala2(x)
        combine_feature = x.view(x.size(0),-1)
        x1 = self.fcout(combine_feature)
        x2 = self.auxiliary2(combine_feature)
        return nn.Sigmoid()(x1), x2


class RN3(nn.Module):
    def __init__(self, C, num_classes):
        super(RN3,self).__init__()

        self.channel_2 = 2 * C
        self.channel_4 = 4 * C
        self.channel_8 = 8 * C
        self.channel_16 = 16 * C

        self.attention3 = nn.Sequential(
            SepConv(
                channel_in= self.channel_8,
                channel_out= self.channel_8
            ),
            nn.BatchNorm2d(self.channel_8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.scala3 = nn.Sequential(
                SepConv(
                    channel_in = self.channel_8,
                    channel_out = self.channel_16
                ),
                nn.AdaptiveAvgPool2d(1)
        )

        self.fcout = nn.Sequential(
                nn.Linear(self.channel_16,self.channel_2),
                nn.BatchNorm1d(self.channel_2),
                nn.ReLU(),
                nn.Linear(self.channel_2,1)
                )

        self.auxiliary3 = nn.Linear(self.channel_16, num_classes)

    def forward(self,x):
        x = self.scala3(x)
        combine_feature = x.view(x.size(0),-1)
        x1 = self.fcout(combine_feature)
        x2 = self.auxiliary3(combine_feature)
        return nn.Sigmoid()(x1), x2


class RN4(nn.Module):
    def __init__(self, C, num_classes):
        super(RN4, self).__init__()

        self.channel_2 = 2 * C
        self.channel_4 = 4 * C
        self.channel_8 = 8 * C
        self.channel_16 = 16 * C

        self.scala4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1)
        )
        self.g_fc1 = nn.Linear(self.channel_16,self.channel_4)
        self.batchNorm1 = nn.BatchNorm1d(self.channel_4)
        self.g_fc2 = nn.Linear(self.channel_4, self.channel_4)
        self.g_fc3 = nn.Linear(self.channel_4, self.channel_4)
        self.g_fc4 = nn.Linear(self.channel_4, self.channel_4)

        self.fcout = nn.Sequential(
                nn.Linear(self.channel_4, 1),
                )

        self.auxiliary4 = nn.Linear(self.channel_16, num_classes)

        self.apply(initialize_weights)


    def forward(self, x):
        """g"""
        x = self.scala4(x)
        combine_feature = x.view(x.size(0),-1)

        x1 = self.g_fc1(combine_feature)
        x1 = self.batchNorm1(x1)
        x1 = F.relu(x1)
        x1 = self.fcout(x1)
        x2 = self.auxiliary4(combine_feature)
        return nn.Sigmoid()(x1), x2

class RN_logits(nn.Module):
    def __init__(self, num_classes):
        super(RN_logits, self).__init__()

        self.logits = nn.Sequential(
                nn.Linear(2*num_classes,1),
                nn.BatchNorm1d(1),
                nn.ReLU(),
                nn.Linear(1,1)
                nn.Sigmoid()
                )
    def forward(self, x):
        x = self.logits(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

def dowmsampleBottleneck(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

class ResNet(nn.Module):

    def __init__(self, args, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.args = args
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.AdaptiveAvgPool2d(1)

        self.attention1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=64 * block.expansion
            ),
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=256 * block.expansion
            ),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.fc4 = nn.Linear(512 * block.expansion, self.args.num_classes)

        #if block is BasicBlock:
            #dim1, dim2, dim3, dim4 = 128,256,512,1024
        #else:
            #dim1, dim2, dim3, dim4 = x,,512,1024

        self.rn1 = RN1(self.args.channel, self.args.num_classes)
        self.rn2 = RN2(self.args.channel, self.args.num_classes)
        self.rn3 = RN3(self.args.channel, self.args.num_classes)
        self.rn4 = RN4(self.args.channel, self.args.num_classes)
        self.rn_logits = RN_logits(self.args.num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        out1_feature = x
        x = self.layer2(x)
        out2_feature = x
        x = self.layer3(x)
        out3_feature = x
        x = self.layer4(x)
        out4_feature = x
        out4_feature_beforeFC = self.scala4(out4_feature).view(x.size(0),-1)
        out4 = self.fc4(out4_feature_beforeFC)
        return out4, [out4_feature, out3_feature, out2_feature, out1_feature]


def _resnet(args, arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(args, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(args, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(args, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(args, pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(args, 'resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(args, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(args, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(args, pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(args, 'resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(args, pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(args, 'resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


