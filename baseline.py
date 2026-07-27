import os

import torch
from torch import nn
import torch.nn.functional as F
import mix_transformer

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class GAM(nn.Module):
    def __init__(self, ch_1, ch_2):  # ch_1:previous, ch_2:current/output
        super(GAM, self).__init__()
        self.ch2 = ch_2
        self.conv_pre = convblock(ch_1, ch_2, 3, 1, 1)

    def forward(self, rgb, pre):
        cur_size = rgb.size()[2:]

        pre = self.conv_pre(F.interpolate(pre, cur_size, mode='bilinear', align_corners=True))

        fus = pre + rgb

        return fus

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # shuffle
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

'''----------CBAM----------'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class CSIM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSIM, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.cbam = CBAM(in_planes=in_ch)
        if in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(out_ch))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, groups=2)
        out = out + residual

        out = self.cbam(out)
        out = self.prelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch),
            ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.block(x)


class EdgeBranch(nn.Module):
    """TensorRT-friendly branch that distills SAM-style edges at 1/4 scale."""

    def __init__(self, in_ch=3, base_ch=16, embed_dim=64):
        super(EdgeBranch, self).__init__()
        mid_ch = base_ch * 2
        self.stem = ConvBNReLU(in_ch, base_ch, kernel_size=3, stride=2, padding=1)
        self.refine1 = DepthwiseSeparableConv(base_ch, base_ch)
        self.down = ConvBNReLU(base_ch, mid_ch, kernel_size=3, stride=2, padding=1)
        self.refine2 = DepthwiseSeparableConv(mid_ch, mid_ch)
        self.refine3 = DepthwiseSeparableConv(mid_ch, mid_ch)
        self.edge_head = nn.Conv2d(mid_ch, 1, kernel_size=1, stride=1, padding=0)
        self.feature_proj = ConvBNReLU(mid_ch, embed_dim, kernel_size=1, stride=1, padding=0)
        self.edge_proj = ConvBNReLU(1, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb):
        x = self.stem(rgb)
        x = self.refine1(x)
        x = self.down(x)
        x = self.refine2(x)
        x = self.refine3(x)
        edge_logit = self.edge_head(x)
        edge_embed = self.feature_proj(x) + self.edge_proj(torch.sigmoid(edge_logit))
        return edge_logit, edge_embed

    def guidance_to_embed(self, edge_guidance):
        return self.edge_proj(edge_guidance)


class Decoder(nn.Module):  # 解码器
    def __init__(self):
        super(Decoder, self).__init__()

        self.d31 = GAM(320, 64)
        self.d42 = GAM(512, 128)
        self.d42_31 = GAM(128, 64)  # 拟合

        self.score_1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.score_2 = nn.Conv2d(128, 1, 1, 1, 0)
        self.score_3 = nn.Conv2d(320, 1, 1, 1, 0)
        self.score_4 = nn.Conv2d(512, 1, 1, 1, 0)

        # FEM
        self.FEM4 = CSIM(in_ch=512, out_ch=512)
        self.FEM3 = CSIM(in_ch=320, out_ch=320)
        self.FEM2 = CSIM(in_ch=128, out_ch=128)
        self.FEM1 = CSIM(in_ch=64, out_ch=64)

    def forward(self, rgb):
        d1, d2, d3, d4 = rgb[0], rgb[1], rgb[2], rgb[3]

        d4 = self.FEM4(d4)
        d3 = self.FEM3(d3)
        d2 = self.FEM2(d2)
        d1 = self.FEM1(d1)

        d13 = self.d31(d1, d3)
        d24 = self.d42(d2, d4)
        d1234 = self.d42_31(d13, d24)

        score1234 = self.score_1(d1234)
        score13 = self.score_1(d13)
        score24 = self.score_2(d24)

        return score1234, score13, score24


class Segformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(mix_transformer, backbone)()
        ## initilize encoder
        if pretrained:
            weight_path = pretrained if isinstance(pretrained, str) else backbone + '.pth'
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(
                    "MiT pretrained weight not found: '{}'. Pass a valid --pretrained "
                    "path, or use --pretrained none to train without ImageNet weights.".format(weight_path)
                )
            state_dict = torch.load(weight_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict.pop('head.weight', None)
            state_dict.pop('head.bias', None)
            self.encoder.load_state_dict(state_dict, strict=False)

    def forward(self):
        model = Segformer('mit_b3', pretrained=True)
        return model


class Mnet(nn.Module):
    def __init__(self, backbone="mit_b3", pretrained=True, edge_channels=16):
        super(Mnet, self).__init__()

        net = Segformer(backbone, pretrained)
        self.rgb_encoder = net.encoder
        edge_embed_dim = self.rgb_encoder.embed_dims[0]
        self.edge_branch = EdgeBranch(base_ch=edge_channels, embed_dim=edge_embed_dim)
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, edge=None, use_teacher_edge=False):
        # rgb
        B = rgb.shape[0]
        rgb_f = []

        edge_logit, edge_embed = self.edge_branch(rgb)

        # stage 1
        x, H, W = self.rgb_encoder.patch_embed1(rgb)
        if use_teacher_edge and edge is not None:
            edge_guidance = F.interpolate(edge, size=(H, W), mode='area')
            edge_embed = self.edge_branch.guidance_to_embed(edge_guidance)
        elif edge_embed.shape[2:] != (H, W):
            edge_embed = F.interpolate(edge_embed, size=(H, W), mode='bilinear', align_corners=True)

        edge_x = edge_embed.flatten(2).transpose(1, 2).contiguous()
        for i, blk in enumerate(self.rgb_encoder.block1):
            x = blk(x, H, W, edge_x)
        x = self.rgb_encoder.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # stage 2
        x, H, W = self.rgb_encoder.patch_embed2(x)
        for i, blk in enumerate(self.rgb_encoder.block2):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # stage 3
        x, H, W = self.rgb_encoder.patch_embed3(x)
        for i, blk in enumerate(self.rgb_encoder.block3):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # stage 4
        x, H, W = self.rgb_encoder.patch_embed4(x)
        for i, blk in enumerate(self.rgb_encoder.block4):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        score1, score2, score3 = self.decoder(rgb_f)

        # return score1
        return score1, score2, score3, self.sigmoid(score1), self.sigmoid(score2), \
               self.sigmoid(score3), edge_logit, self.sigmoid(edge_logit)
