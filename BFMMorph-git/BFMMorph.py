'''
TransMorph model

Swin-Transformer code retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''
import ml_collections
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np
from networks import decoder
import math
from functools import partial





try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
from mamba_ssm import Mamba
from networks import CWA

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Correlation(nn.Module):
    def __init__(self, max_disp=1, kernel_size=1, stride=1, use_checkpoint=False):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert stride == 1, "stride other than 1 is not implemented"
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.max_disp = max_disp
        self.padlayer = nn.ConstantPad3d(max_disp, 0)

    def forward_run(self, x_1, x_2):

        x_2 = self.padlayer(x_2)
        offsetx, offsety, offsetz = torch.meshgrid([torch.arange(0, 2 * self.max_disp + 1),
                                                    torch.arange(0, 2 * self.max_disp + 1),
                                                    torch.arange(0, 2 * self.max_disp + 1)], indexing='ij')

        w, h, d = x_1.shape[2], x_1.shape[3], x_1.shape[4]
        x_out = torch.cat([torch.mean(x_1 * x_2[:, :, dx:dx + w, dy:dy + h, dz:dz + d], 1, keepdim=True)
                           for dx, dy, dz in zip(offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1))], 1)
        return x_out

    def forward(self, x_1, x_2):

        if self.use_checkpoint and x_1.requires_grad and x_2.requires_grad:
            x = checkpoint.checkpoint(self.forward_run, x_1, x_2)
        else:
            x = self.forward_run(x_1, x_2)
        return x
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
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)



class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


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

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """
        x: B, H*W*T, C
        """
        B, C, H, W, T = x.shape
        #assert L == H * W * T, "input feature has wrong size"
        #assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        #x = x.view(B, H, W, T, C)
        x = x.permute(0, 2, 3, 4, 1)
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C
        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, H//2, W//2, T//2, 2 * C)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class FscLayer(nn.Module):
    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FscLayer, self).__init__()
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"
        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        # self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z
        emb = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 64, d_conv = 4, expand = 4, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba1 = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",
                nslices=num_slices,
        )

        self.mlp = FscLayer(input_dim=dim, output_dim=dim, p_ratio=0.25, activation='gelu', use_p_bias=True)#MlpChannel(dim, dim*2)
        self.nslices = num_slices
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat1 = x.reshape(B, C, n_tokens).transpose(-1, -2)
        #x_skip = x_flat1
        x_norm = self.norm(x_flat1)
        xmamba1 = self.mamba1(x_norm) + x_flat1
        x_norm2 = self.norm2(xmamba1)
        out = self.mlp(x_norm2) + xmamba1
        out = out.transpose(-1, -2).reshape(B, C, *img_dims)

        return out


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.pos_em = SinPositionalEncoding3D(96).cuda()
        self.patch_em = PatchEmbed(patch_size=4, in_chans=in_chans, embed_dim=96,norm_layer=nn.LayerNorm)
        self.downsample_layers.append(self.patch_em)
        for i in range(2):
            downsample_layer = PatchMerging(dim=dims[i], norm_layer=nn.LayerNorm, reduce_factor=4)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        num_slices_list = [40, 20, 10]
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        '''for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)'''

    def forward_features(self, x):
        outs = []

        for i in range(3):
            x = self.downsample_layers[i](x)
            if i == 0:
                x = (x + self.pos_em(x))

            x = self.stages[i](x)

            '''if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)'''
            outs.append(x)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class MambamatchEncoder(nn.Module):
    def __init__(self, depths = 2, dims = 96, num_slices = 40):
        super().__init__()
        self.stage1 = []
        self.norm = nn.InstanceNorm3d(dims).cuda()
        for i in range(depths):
            self.stage1.append(MambaLayer(dim=dims, num_slices=num_slices).cuda())
        self.depth = depths
    def forward_features(self, x):
        outs = []
        x_out = x
        for i in range(self.depth):
            x_out = self.stage1[i](x_out)
        #x_out = self.norm(x_out)
        outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, dim, norm=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))
        self.norm = norm(dim)
    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        feat = feat.permute(0, 4, 1, 2, 3)
        return feat

class BFMMorph(nn.Module):
    def __init__(self, config):
        '''
        TransMorph Model
        '''
        super(BFMMorph, self).__init__()
        self.transformer1 = MambaEncoder(in_chans=1,
                                depths=[1,1,2],
                                dims=[96, 192, 384],
                                drop_path_rate=0,
                              )
        self.transformer2 = []
        self.conlist = []
        self.cwa = []
        self.insnorm = []
        self.Corr = Correlation(max_disp=1, use_checkpoint=False)
        num_slices_list = [40 , 20, 10]
        de = [1,1,2]
        for i in range(3):
            self.transformer2.append(MambamatchEncoder(depths=de[i],
                                dims=96*(2**i),
                                num_slices = num_slices_list[i]
                              ).cuda())
            self.conlist.append(nn.Conv3d(96*(2**(i+1)),96*(2**i),kernel_size=3, stride=1, padding='same').cuda())
            self.cwa.append(CWA(96*(2**i)).cuda())
            self.insnorm.append(nn.InstanceNorm3d(96*(2**i)).cuda())
        self.EMCAD = decoder()
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.LR = nn.LeakyReLU()
    def forward(self, source, target):
        sourcelist = self.transformer1(source)
        targetlist = self.transformer1(target)
        #out_feats1 = self.transformer1(torch.cat([source, target], dim=1))
        #torch.Size([1, 96, 40, 48, 40])
        #torch.Size([1, 192, 20, 24, 20])
        #torch.Size([1, 384, 10, 12, 10])
        #torch.Size([1, 768, 5, 6, 5])
        out_feats1 = []
        for i in range(3):
            x_corr = self.Corr(sourcelist[i],targetlist[i])
            x = torch.cat([sourcelist[i], targetlist[i]], dim=1)
            x = self.conlist[i](x)
            x = self.insnorm[i](x)
            x = self.LR(x)
            x = self.cwa[i](x)*x
            x = self.transformer2[i](x)[0]
            x = torch.cat([x,  x_corr], dim=1)
            out_feats1.append(x)


        flow = self.decoder(source, target, out_feats1)
        moved = self.spatial_trans(source, flow)
        return moved, flow

def get_3DBFMMorph_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 1
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)  # (2, 2, 4, 2)
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = True
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)  # (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160,192,224)
    config.d_state = 16
    config.d_conv = 4
    config.expand = 2
    return config

# from TransMorph import CONFIGS as CONFIGS_TM
if __name__ =='__main__':
    config = get_3DBFMMorph_config()
    model = BFMMorph(config=config).to('cuda')
    source = torch.randn(1,1,160,192,224).cuda()
    target = torch.randn(1,1,160,192,224).cuda()
    output = model(source, target)
    print(output[0].shape)
    print(output[1].shape)
