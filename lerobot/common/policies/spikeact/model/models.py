# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based.layer import Dropout
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

decay = 0.25


class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  # change the max value

    def forward(self, x):

        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        return output


class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)


class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.lif1 = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1))
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim)
        # self.Conv = MHMC(dim=dim)

        self.lif1 = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv1 = RepConv(dim, dim*mlp_ratio)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.lif2 = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, 4 * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = LIFNode(
            tau=2.0, step_mode="m", detach_reset=True, backend="cupy"
        )

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = LIFNode(
            tau=2.0, step_mode="m", detach_reset=True, backend="cupy"
        )
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = LIFNode(
            tau=2.0, step_mode="m", detach_reset=True, backend="cupy"
        )

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")

        self.k_lif = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")

        self.v_lif = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")

        self.attn_lif = LIFNode(
            tau=2.0, v_threshold=0.5, step_mode="m", detach_reset=True, backend="cupy"
        )

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class MS_MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        sr_ratio=1,
        kdim=None,
        vdim=None,
    ):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"dim {embed_dim} should be divided by num_heads {num_heads}."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        # self.scale = 0.125
        self.scale = 0.125 / 64

        # self.head_lif_q = LIFNode(
        #     tau=2.0, step_mode="m", detach_reset=True, backend="cupy"
        # )
        # self.head_lif_k = LIFNode(
        #     tau=2.0, step_mode="m", detach_reset=True, backend="cupy"
        # )
        # self.head_lif_v = LIFNode(
        #     tau=2.0, step_mode="m", detach_reset=True, backend="cupy"
        # )

        self.head_lif_q = mem_update()
        self.head_lif_k = mem_update()
        self.head_lif_v = mem_update()

        if self._qkv_same_embed_dim:

            self.q_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False),
                Transpose(-1, -2),
                nn.BatchNorm1d(embed_dim),
                Transpose(-1, -2),
            )

            self.k_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False),
                Transpose(-1, -2),
                nn.BatchNorm1d(embed_dim),
                Transpose(-1, -2),
            )

            self.v_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False),
                Transpose(-1, -2),
                nn.BatchNorm1d(embed_dim),
                Transpose(-1, -2),
            )

        else:
            self.q_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False),
                Transpose(-1, -2),
                nn.BatchNorm1d(embed_dim),
                Transpose(-1, -2),
            )

            self.k_proj = nn.Sequential(
                nn.Linear(kdim, embed_dim, bias=False),
                Transpose(-1, -2),
                nn.BatchNorm1d(embed_dim),
                Transpose(-1, -2),
            )

            self.v_proj = nn.Sequential(
                nn.Linear(vdim, embed_dim, bias=False),
                Transpose(-1, -2),
                nn.BatchNorm1d(embed_dim),
                Transpose(-1, -2),
            )

        # self.q_lif = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")

        # self.k_lif = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")

        # self.v_lif = LIFNode(tau=2.0, step_mode="m", detach_reset=True, backend="cupy")

        self.q_lif = mem_update()

        self.k_lif = mem_update()

        self.v_lif = mem_update()

        # self.attn_lif = LIFNode(
        #     tau=2.0, v_threshold=0.5, step_mode="m", detach_reset=True, backend="cupy"
        # )

        self.attn_lif = mem_update()

        self.dropout = Dropout(dropout, step_mode="m")
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            Transpose(-1, -2),
            nn.BatchNorm1d(embed_dim),
            Transpose(-1, -2),
        )

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,  # (B, S)
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):

        T, B, L, Eq = query.shape
        T, B, S, Ek = key.shape
        T, B, S, Ev = value.shape

        q = self.head_lif_q(query)
        k = self.head_lif_k(key)
        v = self.head_lif_v(value)

        q = self.q_proj(q.flatten(0, 1)).reshape(T, B, L, -1)
        k = self.k_proj(k.flatten(0, 1)).reshape(T, B, S, -1)
        v = self.v_proj(v.flatten(0, 1)).reshape(T, B, S, -1)

        q = self.q_lif(q)
        q = (
            q.reshape(T, B, L, self.num_heads, self.embed_dim // self.num_heads)
            .permute(0, 1, 3, 2, 4)  # (T, B, head, L, emb/head)
            .contiguous()
        )

        k = self.k_lif(k)
        k = (
            k.reshape(T, B, S, self.num_heads, self.embed_dim // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v)
        v = (
            v.reshape(T, B, S, self.num_heads, self.embed_dim // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # x = k.transpose(-2, -1) @ v  # (T, B, head, emb/head, emb/head)
        # x = (q @ x) * self.scale  # (T, B, head, L, emb/head)

        x = q @ k.transpose(-2, -1)  # (T, B, head, L, S)
        if key_padding_mask is not None:
            x = torch.masked_fill(
                x, key_padding_mask.unsqueeze(0).unsqueeze(2).unsqueeze(2), 0
            )
        x = self.dropout(x)
        x = (x @ v) * self.scale  # (T, B, head, L, emb/head)

        x = (
            x.transpose(3, 4)
            .reshape(T, B, self.embed_dim, L)
            .transpose(2, 3)
            .contiguous()
        )

        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.proj(x).reshape(T, B, L, self.embed_dim)

        return x
