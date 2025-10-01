import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


# def calc_mean_std(features):
#     """
#     :param features: shape of features -> [batch_size, c, h, w]
#     :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
#     """

#     batch_size, c = features.size()[:2]
#     features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
#     features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
#     return features_mean, features_std

# from StyleGaussian
def calc_mean_std(x, eps=1e-8):
    """
    calculating channel-wise instance mean and standard variance
    x: shape of (N,C,*)
    """
    mean = torch.mean(x.flatten(2), dim=-1, keepdim=True) # size of (N, C, 1)
    std = torch.std(x.flatten(2), dim=-1, keepdim=True) + eps # size of (N, C, 1)
    
    return mean, std

def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    b, c, _, _ = content_features.shape
    
    content_mean, content_std = calc_mean_std(content_features)
    content_mean = content_mean.reshape(b, c, 1, 1)
    content_std = content_std.reshape(b, c, 1, 1)
    
    style_mean, style_std = calc_mean_std(style_features)
    style_mean = style_mean.reshape(b, c, 1, 1)
    style_std = style_std.reshape(b, c, 1, 1)
    
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    
    return normalized_features

def AdaIN_pcd(content_features, style_features):
    """apply AdaIN on the whole pointcloud altogether (refer to StyleGaussian)
    or view-wise
    
    Input:
        content_features [b, c, n] / [b, c, h1, w1]
        style_features [b, c, h2, w2]
    
    Output:
        normalized_features [b, c, n] / [b, c, h1, w1]
    """
    if content_features.dim() == 4:
        output_dim = 4
        b, c, h1, w1 = content_features.shape
        content_features = content_features.reshape(b, c, -1)
    else:
        output_dim = 3
    
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
        
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    
    if output_dim == 4:
        normalized_features = normalized_features.reshape(b, c, h1, w1)
    
    return normalized_features

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        
        self.requires_grad_(False)

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4


class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h


class AdaIN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()

    def generate(self, content_images, style_images, alpha=1.0):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)
        return out

    @staticmethod
    def calc_content_loss(out_features, t):
        return F.mse_loss(out_features, t)

    @staticmethod
    def calc_style_loss(content_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def forward(self, content_images, style_images, alpha=1.0, lam=10):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)

        output_features = self.vgg_encoder(out, output_last_feature=True)
        output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)

        loss_c = self.calc_content_loss(output_features, t)
        loss_s = self.calc_style_loss(output_middle_features, style_middle_features)
        loss = loss_c + lam * loss_s
        return loss