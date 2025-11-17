import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from .common import DropPath, Identity

# 设置随机种子
seed = 317
seed = 714
torch.manual_seed(seed)  # 固定CPU随机性
torch.cuda.manual_seed(seed)  # 固定当前GPU随机性
torch.cuda.manual_seed_all(seed)  # 固定所有GPU的随机性（多GPU）


# 卷积操作 + batchnorm + 激活
class ConvBNLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
            groups=1,  # 分组卷积的组数
            act=nn.GELU,
    ):
        super().__init__()  # 继承父类
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)  # batchnorm
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class DADCAttention(nn.Module):

    def __init__(self, in_channels, out_channels, local_k, inter_channels, num_heads=8):
        super(DADCAttention, self).__init__()
        assert out_channels % num_heads == 0, \
            f"out_channels ({out_channels}) must be a multiple of num_heads ({num_heads})"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.local_k = local_k
        self.num_heads = num_heads
        self.norm = nn.BatchNorm2d(in_channels)

        # Convolutional kernels for horizontal and vertical attention
        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, local_k, 1))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, local_k))
        # 多头空间注意力卷积层
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.fused_conv = nn.Conv2d(2 * self.out_channels, self.out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.kv, std=0.001)
        nn.init.trunc_normal_(self.kv3, std=0.001)
        nn.init.kaiming_normal_(self.spatial_conv.weight, mode='fan_out')  # 初始化空间卷积
        nn.init.kaiming_normal_(self.fused_conv.weight, mode='fan_out')

    def _act_dn(self, x):
        n, c, h, w = x.size()
        # 分割多头 [n, num_heads, c_per_head, h, w]
        x_multihead = x.view(n, self.num_heads, c // self.num_heads, h, w)

        # 计算每个头的空间注意力
        avg_pool = x_multihead.mean(dim=2, keepdim=True)  # [n, num_heads, 1, h, w]
        max_pool = x_multihead.max(dim=2, keepdim=True)[0]  # [n, num_heads, 1, h, w]
        spatial_attn = torch.cat([avg_pool, max_pool], dim=2)  # [n, num_heads, 2, h, w]

        # 合并批次和多头维度以进行卷积
        spatial_attn = spatial_attn.flatten(0, 1)  # [n*num_heads, 2, h, w]
        spatial_attn = self.spatial_conv(spatial_attn)  # [n*num_heads, 1, h, w]

        # 恢复多头维度并激活
        spatial_attn = spatial_attn.view(n, self.num_heads, 1, h, w)
        spatial_attn = torch.sigmoid(spatial_attn)

        # 应用注意力权重并合并多头
        x_multihead = x_multihead * spatial_attn  # [n, num_heads, c//h, h, w]
        return x_multihead.view(n, c, h, w)

    def forward(self, x):
        x = self.norm(x)
        # Vertical attention
        x1 = F.conv2d(x, self.kv, bias=None, stride=1,
                      padding=((self.local_k - 1) // 2, 0))  # [b, inter_channels, h, w]
        x1 = self._act_dn(x1)  # [b, inter_channels, h, w]
        x1 = F.conv2d(x1, self.kv.transpose(1, 0), bias=None, stride=1,
                      padding=((self.local_k - 1) // 2, 0))  # [b, in_channels, h, w]

        # Horizontal attention
        x3 = F.conv2d(x, self.kv3, bias=None, stride=1, padding=(0, (self.local_k - 1) // 2))
        x3 = self._act_dn(x3)
        x3 = F.conv2d(x3, self.kv3.transpose(1, 0), bias=None, stride=1, padding=(0, (self.local_k - 1) // 2))

        fused = torch.cat([x1, x3], dim=1)
        fused = self.fused_conv(fused)

        return fused


class DPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, local_k, num_heads=8, HW=None):
        super(DPBlock, self).__init__()
        self.attn_l = DADCAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            local_k=local_k,
            inter_channels=96,
            num_heads=num_heads
        )
        self.HW = HW

    def forward(self, x):
        x = self.attn_l(x)  # (B, C, H, W)  192,8,32

        return x


class Mlp(nn.Module):

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 2).reshape(-1, c, h, w)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """输入形状: (B, C, H, W)
           输出形状: (B, C, H, W)
        """
        B, C, H, W = x.size()

        # 计算通道统计量（沿空间维度H和W）
        avg_pool = x.mean(dim=(2, 3))  # (B, C)

        # 通道注意力计算
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))  # (B, C)

        # 合并并生成注意力权重
        channel_att = self.sigmoid(avg_out)  # (B, C)

        # 扩展维度以便广播
        channel_att = channel_att.view(B, C, 1, 1)

        # 应用通道注意力
        return x * channel_att


class PositionAttention(nn.Module):
    def __init__(self, dim, inter_channels, hw):
        super().__init__()
        self.dim = dim
        self.num_channels = inter_channels
        self.conv1 = nn.Conv2d(in_channels=dim, kernel_size=1, out_channels=inter_channels)
        self.conv2 = nn.Conv2d(in_channels=dim, kernel_size=1, out_channels=dim)
        self.pre_weights = nn.Parameter(torch.ones(1, 1, hw[0], hw[1]))
        self.post_weights = nn.Parameter(torch.ones(1, 1, hw[0], hw[1]))

    def forward(self, x):
        n, d, h, w = x.size()
        x = x * self.pre_weights

        q = self.conv1(x).flatten(2, 3).transpose(1, 2)  # n, hw, num_channels
        k = self.conv1(x).flatten(2, 3)  # n, num_channels, hw
        v = self.conv2(x).flatten(2, 3)  # n, d, hw
        attn_scores = torch.bmm(q, k)  # n, hw, hw
        attn_scores = attn_scores / (d ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_vecs = torch.bmm(attn_scores, v.transpose(1, 2))  # n, hw, d
        attn_vecs = attn_vecs.transpose(1, 2).reshape(n, d, h, w)

        attn_vecs = attn_vecs * self.post_weights

        return attn_vecs


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mixer='DPB',
            local_mixer=[],
            HW=None,
            mlp_ratio=4.0,  # MLP的隐藏层维度相对于输入维度的比例
            drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer='nn.LayerNorm',
            eps=1e-6,
            prenorm=True,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            # eval(norm_layer) 会将字符串 norm_layer 作为一个 Python 表达式执行
            self.norm1 = eval(norm_layer)(dim, eps=eps)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'DPB':
            self.mixer = DPBlock(in_channels=dim,
                                 out_channels=dim,
                                 local_k=local_mixer,
                                 num_heads=num_heads,
                                 HW=HW)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv, DPB]')

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=eps)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.prenorm = prenorm

    def forward(self, x):
        # 残差连接
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
            self,
            img_size=[32, 128],
            in_channels=3,
            embed_dim=768,
            sub_num=2,  # 控制层数，指定使用几层卷积来提取特征
            patch_size=[4, 4],
            mode='pope',
    ):
        super().__init__()
        num_patches = (img_size[1] // (2 ** sub_num)) * (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm_1 = nn.BatchNorm2d(embed_dim // 2)
        self.norm_2 = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(
            in_channels=embed_dim * 3,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv1_1 = nn.Conv2d(
            in_channels=embed_dim // 2,
            out_channels=embed_dim // 4,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=embed_dim // 3,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv1_3 = nn.Conv2d(
            in_channels=embed_dim // 2,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv1_4 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim // 3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.global_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=None,
                    ),
                )
                self.proj_3 = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=None,
                    )
                )
                self.proj_3_s = ConvBNLayer(
                    in_channels=embed_dim // 4,
                    out_channels=embed_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU
                )
                self.proj_5_s = ConvBNLayer(
                    in_channels=embed_dim // 4,
                    out_channels=embed_dim // 2,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    act=nn.GELU
                )
                self.proj_7_s = ConvBNLayer(
                    in_channels=embed_dim // 4,
                    out_channels=embed_dim // 2,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    act=nn.GELU
                )

        elif mode == 'linear':
            self.proj = nn.Conv2d(3,
                                  embed_dim,
                                  kernel_size=patch_size,
                                  stride=patch_size)
            self.num_patches = img_size[0] // patch_size[0] * img_size[
                1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
                H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        '''
        transformer 等模型需要处理的格式是 [batch_size, num_patches, embed_dim]
        '''
        x = self.proj_3(x)  # 96,16,64
        x1 = self.proj(x)
        x_ = self.conv1_1(x)  # 48,16,64

        x2 = self.proj_5_s(x_)  # 96,8,32
        x3 = self.proj_7_s(x_)  # 96,8,32

        xm = torch.concat((x2, x3), dim=1)  # 192,8,32

        # 计算权重
        x_pool = self.global_pool(xm)  # 192,8,32
        x_pool = self.conv1_4(x_pool)  # 64,8,32
        x_pool = self.conv1_2(self.act(x_pool))  # 192,8,32
        prob = self.sigmoid(x_pool)

        # 将prob分配给每个分支
        prob_x2 = prob[:, :x2.size(1), :, :]  # [N, C2, H, W]  96,8,32
        prob_x3 = prob[:, -x2.size(1):, :, :]  # [N, C3, H, W]  96,8,32

        # 逐元素相乘加权
        weighted_x2 = x2 * prob_x2  # [N, C2, H, W]  96,8,32
        weighted_x3 = x3 * prob_x3  # [N, C3, H, W]  96,8,32
        fused_feature = weighted_x2 + weighted_x3  # 96,8,32
        fused_feature = self.conv1_3(fused_feature)  # 192,8,32

        # 最终融合结果
        x = self.act(fused_feature + x1)  # 192,8,32
        # x = x.flatten(2).transpose(1, 2)  # 256, 192

        return x


class SubSample(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            types='Pool',
            stride=[2, 1],
            sub_norm='nn.LayerNorm',
            act=None,
    ):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(kernel_size=[3, 5],
                                        stride=stride,
                                        padding=[1, 2])
            self.maxpool = nn.MaxPool2d(kernel_size=[3, 5],
                                        stride=stride,
                                        padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=1)
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        if self.types == 'Pool':
            x1 = self.avgpool(x)  # [B, C, 16, 128] imgsize:[32, 128]
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).transpose(1, 2))  # [B, 16*128, C]
        else:
            x = self.conv(x)
            # out = x.flatten(2).transpose(1, 2)
        out = self.norm(x)
        if self.act is not None:
            out = self.act(out)

        return out


class SCC3(nn.Module):

    def __init__(
            self,
            img_size=[32, 100],
            in_channels=1,
            embed_dim=[64, 128, 256],
            depth=[3, 6, 3],  # block 的数量
            num_heads=[2, 4, 8],
            mixer=['Local'] * 6 + ['Global'] * 6,  # Local atten, Global atten, Conv
            local_mixer=[3, 5, 7],
            patch_merging='Conv',  # Conv, Pool, None
            sub_k=[[2, 1], [2, 1]],
            mlp_ratio=4,
            drop_rate=0.0,
            last_drop=0.1,
            drop_path_rate=0.1,
            norm_layer='nn.BatchNorm2d',
            sub_norm='nn.BatchNorm2d',
            eps=1e-6,
            out_channels=192,
            out_char_num=25,
            block_unit='Block',
            act='nn.GELU',
            last_stage=True,
            sub_num=2,
            prenorm=True,
            use_lenhead=False,
            feature2d=False,
            **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        self.feature2d = feature2d
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num,
        )
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2 ** sub_num), img_size[1] // (2 ** sub_num)]
        self.hw = [
            [self.HW[0] // sub_k[0][0], self.HW[1] // sub_k[0][1]],
            [self.HW[0] // (sub_k[0][0] * sub_k[1][0]), self.HW[1] // (sub_k[0][1] * sub_k[1][1])],
        ]

        Block_unit = eval(block_unit)

        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mixer=mixer[0:depth[0]][i],
                HW=self.HW,
                local_mixer=local_mixer[0],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                act_layer=eval(act),
                drop_path=dpr[0:depth[0]][i],
                norm_layer=norm_layer,
                eps=eps,
                prenorm=prenorm,
            ) for i in range(depth[0])
        ])
        self.pos_att1 = PositionAttention(dim=embed_dim[0], inter_channels=embed_dim[0] // 8, hw=self.HW)
        self.channel_att1 = ChannelAttention(in_channels=embed_dim[0], reduction=4)
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=sub_k[0],
                types=patch_merging,
            )
            HW = self.hw[0]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mixer=mixer[depth[0]:depth[0] + depth[1]][i],
                HW=HW,
                local_mixer=local_mixer[1],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                act_layer=eval(act),
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                norm_layer=norm_layer,
                eps=eps,
                prenorm=prenorm,
            ) for i in range(depth[1])
        ])
        self.pos_att2 = PositionAttention(dim=embed_dim[1], inter_channels=embed_dim[1] // 8, hw=self.hw[0])
        self.channel_att2 = ChannelAttention(in_channels=embed_dim[1], reduction=4)
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=sub_k[1],
                types=patch_merging,
            )
            HW = self.hw[1]
        self.blocks3 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mixer=mixer[depth[0] + depth[1]:][i],
                HW=HW,
                local_mixer=local_mixer[2],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                act_layer=eval(act),
                drop_path=dpr[depth[0] + depth[1]:][i],
                norm_layer=norm_layer,
                eps=eps,
                prenorm=prenorm,
            ) for i in range(depth[2])
        ])
        self.pos_att3 = PositionAttention(dim=embed_dim[2], inter_channels=embed_dim[2] // 8, hw=self.hw[1])
        self.channel_att3 = ChannelAttention(in_channels=embed_dim[2], reduction=4)
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop)
        else:
            self.out_channels = embed_dim[2]
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], eps=eps)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(p=last_drop)

        self.apply(self._init_weights)

    # 权重初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    # 返回一个包含不需要权重衰减的模型参数名称的集合
    def no_weight_decay(self):
        return {'pos_embed', 'sub_sample1', 'sub_sample2', 'sub_sample3'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks1:
            x = blk(x)

        x = x + self.channel_att1(self.pos_att1(x))

        if self.patch_merging is not None:
            x = self.sub_sample1(x)
        for blk in self.blocks2:
            x = blk(x)
        x = x + self.channel_att2(self.pos_att2(x))

        if self.patch_merging is not None:
            x = self.sub_sample2(x)
        for blk in self.blocks3:
            x = blk(x)
        x = x + self.channel_att3(self.pos_att3(x))

        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.feature2d:
            x = x.transpose(1, 2).reshape(-1, self.embed_dim[2], self.hw[1][0],
                                          self.hw[1][1])
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            x = self.avg_pool(
                x.transpose(1, 2).reshape(-1, self.embed_dim[2], self.hw[1][0],
                                          self.hw[1][1]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
            # return x
            x = x.flatten(2).transpose(1, 2)
        if self.use_lenhead:
            return x, len_x
        # [B, out_char_num, out_channels]

        return x
