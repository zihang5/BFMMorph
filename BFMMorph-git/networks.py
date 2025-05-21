import torch
import torch.nn as nn
# from decodersprelu import EMCAD
import torch.nn.functional as F
from torch.distributions.normal import Normal
# from dynamic3D import DySample
from timm.models.helpers import named_apply
from functools import partial
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU()
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class newEUCB(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class CWA(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CWA, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv3d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv3d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

    #   Spatial attention block (SAB)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class decoder(nn.Module):
    def __init__(self, start_channel=4, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True,
                 lgag_ks=3, activation='relu', encoder='resnet18', pretrain=False):
        super(decoder, self).__init__()
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)
        c_corr = 27

        self.up1 = newEUCB(384+c_corr, 192, skip_channels=192+c_corr, use_batchnorm=False)
        self.up2 = newEUCB(192, 96, skip_channels=96+c_corr, use_batchnorm=False)
        self.up3 = newEUCB(96, 48, skip_channels=48, use_batchnorm=False)
        self.up4 = newEUCB(48, 16, skip_channels=16, use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.reg_head1 = RegistrationHead(
            in_channels=96,
            out_channels=3,
            kernel_size=3,
        )
        self.reg_head2 = RegistrationHead(
        in_channels=48,
        out_channels=3,
        kernel_size=3,
        )
        self.reg_head3 = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        inshape = [160,192,224]
        self.transformer = nn.ModuleList()
        for i in range(2):
            self.transformer.append(SpatialTransformer([s // 2 ** i for s in inshape]))
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True)
    def forward(self, moving_Input, fixed_Input, x):
        input_fusion = torch.cat((moving_Input, fixed_Input), dim=1)
        f5 = self.c2(input_fusion)
        x_s1 = self.avg_pool(input_fusion)  # 用于concat AB后下采样1/2后的卷积的input

        f4 = self.c1(x_s1)

        # if grayscale input, convert to 3 channels
        # encoder
        x1, x2, x3 = x
        #x = self.up0(x4, x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)  # 86
        w = self.reg_head1(x)
        flow = self.upsample_trilin(2 * w)

        x = self.up3(x, f4)
        w = self.reg_head2(x)
        flow = self.upsample_trilin(2 * (self.transformer[1](flow, w) + w))

        x = self.up4(x, f5)
        w = self.reg_head3(x)
        outputs = self.transformer[0](flow, w) + w

        return outputs


if __name__ == '__main__':
    model = decoder().cuda()
    # input_tensor = torch.randn(1, 3, 160, 192, 160).cuda()
    input_tensor = [torch.randn(1, 96 * (2 ** (i - 1)), 80 // (2 ** i), 96 // (2 ** i), 80 // (2 ** i)).cuda() for i in
                    range(1, 5)]
    P = model(input_tensor)
    print(P.shape)

